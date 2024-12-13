import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import glob
# from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from tqdm import tqdm
from tensorboardX import SummaryWriter
import json
import copy

# datasets and splits
from datasets.loader_downstream import MoleculeDataset
from datasets.loader_pretrain import mol_frag_collate
from datasets.utils.splitters import scaffold_split, random_split, random_scaffold_split

# models
from models.mamba_fuser import MSE
# utils
from config_reg import cfg
from datasets.utils.logger import setup_logger, set_seed, dict_to_markdown


if __name__ == '__main__':
    set_seed(cfg.runseed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    # with open(os.path.join(cfg.save_dir, 'config.json'), 'w') as f:
    #     json.dump(args, f, indent=4)

    logger = setup_logger(f"{cfg.model}", cfg.save_dir)
    logger.info('-' * 60)
    logger.info(f'The setup args are:\n{dict_to_markdown(cfg.to_dict(), max_str_len=120)}')

    cfg.save_yaml(filename=cfg.save_dir + 'config.yaml')

    for filename in glob.glob(os.path.join(cfg.save_dir, 'events.out.*')):
        os.remove(filename)

    for filename in glob.glob(os.path.join(cfg.save_dir, f'{cfg.model}_*')):
        os.remove(filename)

    writer = SummaryWriter(log_dir=cfg.save_dir, comment=f'cfg.dataset.lower()',
                           filename_suffix=f'{cfg.dataset.lower()}_{cfg.model.lower()}')

    # logger.info('The setup args are:\n' + f'{dict_to_markdown(cfg, max_str_len=120)}')

    # load dataset
    dataset = MoleculeDataset(dataset=cfg.dataset, root=os.path.join(cfg.data_dir, cfg.dataset.lower()))

    smiles_list = pd.read_csv(os.path.join(cfg.data_dir, cfg.dataset.lower(), 'processed', 'smiles.csv'), header=None)[
        0].tolist()

    # train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list,
    #                                                             null_value=0,
    #                                                             frac_train=0.8,
    #                                                             frac_valid=0.1,
    #                                                             frac_test=0.1)
    if cfg.split == 'scaffold':
        smiles_list = pd.read_csv(cfg.data_dir + cfg.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        logger.info('split via scaffold')
    elif cfg.split == 'random':
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=cfg.seed)
        logger.info('randomly split')
    elif cfg.split == 'random_scaffold':
        smiles_list = pd.read_csv(cfg.data_dir + cfg.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=cfg.seed)
        logger.info('random scaffold')
    else:
        raise ValueError('Invalid split option.')

    num_tasks = train_dataset[0].y.shape[-1]

    logger.info(
        f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}, Test samples: {len(test_dataset)}")
    logger.info(f"Number of tasks: {num_tasks}")

    trainloader = DataLoader(train_dataset, batch_size=cfg.batchsize, collate_fn=mol_frag_collate, shuffle=True,
                             num_workers=0, drop_last=True)

    valloader = DataLoader(valid_dataset, batch_size=cfg.batchsize, collate_fn=mol_frag_collate, shuffle=False,
                           num_workers=0, drop_last=True)

    testloader = DataLoader(test_dataset, batch_size=cfg.batchsize, collate_fn=mol_frag_collate, shuffle=False,
                            num_workers=0, drop_last=True)

    model = MSE(cfg, num_tasks=num_tasks)

    if cfg.pretrain_path:
        check_points = torch.load(cfg.pretrain_path, map_location=device)
        if 'gnn' in check_points.keys():
            model.gnn.load_state_dict(check_points['gnn'])
        else:
            model.gnn.load_state_dict(check_points['mol_gnn'])

    model.to(device)
    best_model = model
    logger.info(f"Model: {cfg.model}, Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    # for name, module in model.named_children():
    #     total_params = sum(p.numel() for p in module.parameters()) / 1e6
    #     logger.info(f"{name}: {total_params:.2f}M")

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    criterion1 = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)

    best_val = 10000.
    best_test = None
    total_steps = len(trainloader)
    for epoch in range(cfg.epoch):
        model.train()
        cum_loss = 0
        cum_loss_frag_div = 0
        cum_loss_frag = 0
        cum_loss_tree = 0
        cum_loss_task = 0
        cum_loss_mask = 0
        for step, batch in enumerate(trainloader):

            batch = batch.to(device)
            pred, loss, losses = model(batch)

            loss_frag_div, loss_frag, loss_tree, loss_mask, loss_task = losses

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            cum_loss += float(loss.cpu().item())
            cum_loss_frag_div += float(loss_frag_div.cpu().item())
            cum_loss_frag += float(loss_frag.cpu().item())
            cum_loss_tree += float(loss_tree.cpu().item())
            cum_loss_mask += float(loss_mask.cpu().item())
            cum_loss_task += float(loss_task.cpu().item())

            # if step % 100 == 1:
            #     logger.info(f"Epoch {epoch}: {step}/{total_steps}, Loss: {cum_loss / (step + 1):.4f}, "
            #                 f"Loss_frag_div: {cum_loss_frag_div / (step + 1):.4f}, "
            #                 f"Loss_frag: {cum_loss_frag / (step + 1):.4f}, "
            #                 f"Loss_tree: {cum_loss_tree / (step + 1):.4f}, "
            #                 f"Loss_mask: {cum_loss_mask / (step + 1):.4f}, "
            #                 f"Loss_task: {cum_loss_task / (step + 1):.4f}")

        # VAL
        model.eval()
        y_pred = []
        y_true = []
        for step, batch in enumerate(valloader):
            # batch.y[batch.y == -1] = 0
            batch = batch.to(device)
            with torch.no_grad():
                pred, loss, losses = model(batch)

                loss_frag_div, loss_frag, loss_tree, loss_mask, loss_task = losses

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        val_res = torch.sqrt(criterion1(y_pred, y_true))

        # TEST
        model.eval()
        y_pred = []
        y_true = []
        for step, batch in enumerate(testloader):
            # batch.y[batch.y == -1] = 0
            batch = batch.to(device)
            with torch.no_grad():
                pred, loss, losses = model(batch)

                loss_frag_div, loss_frag, loss_tree, loss_mask, loss_task = losses

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        test_res = torch.sqrt(criterion1(y_pred, y_true))

        if val_res < best_val:
            best_val = val_res
            best_test = test_res

        logger.info(
            f"Epoch: {epoch:03d}/{cfg.epoch:03d}, Total Loss: {cum_loss:.4f}, "
            f"loss_frag_div: {cum_loss_frag_div:.4f}, loss_frag: {cum_loss_frag:.4f}, "
            f"loss_tree: {cum_loss_tree:.4f}, loss_mask:{cum_loss_mask:.4f}, loss_task: {cum_loss_task:.4f},"
            f"Val RMSE: {val_res:.4f}, Test RMSE: {test_res:.4f}"
        )

    logger.info(f"Best Val RMSE: {best_val.item()}, \tBest Test RMSE: {best_test.item()}")
