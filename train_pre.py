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
from datasets.utils.data_utils import mol_frag_collate
from datasets.utils.splitters import scaffold_split, random_split, random_scaffold_split

# models
from models.mamba_fuser import MSE
# utils
from config import cfg
from datasets.utils.logger import setup_logger, set_seed, dict_to_markdown


if __name__ == '__main__':
    set_seed(cfg.runseed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    # 打包代码
    # os.system('zip -r code.zip *.py')

    logger = setup_logger(f"{cfg.model}", cfg.save_dir)
    logger.info('-' * 60)
    logger.info(f'The setup args are:\n{dict_to_markdown(cfg.to_dict(), max_str_len=120)}')

    cfg.save_yaml(filename=cfg.save_dir + 'config.yaml')

    for filename in glob.glob(os.path.join(cfg.save_dir, 'events.out.*')):
        os.remove(filename)

    # for filename in glob.glob(os.path.join(cfg.save_dir, f'{cfg.model}_*')):
    #     os.remove(filename)

    writer = SummaryWriter(log_dir=cfg.save_dir, comment=f'cfg.dataset.lower()',
                           filename_suffix=f'{cfg.dataset.lower()}_{cfg.model.lower()}')

    # logger.info('The setup args are:\n' + f'{dict_to_markdown(cfg, max_str_len=120)}')

    # load dataset
    dataset = MoleculeDataset(dataset=cfg.dataset, root=os.path.join(cfg.data_dir, cfg.dataset.lower()))

    smiles_list = pd.read_csv(os.path.join(cfg.data_dir, cfg.dataset.lower(), 'processed', 'smiles.csv'), header=None)[0].tolist()

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

    logger.info(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}, Test samples: {len(test_dataset)}")
    logger.info(f"Number of tasks: {num_tasks}")

    trainloader = DataLoader(train_dataset, batch_size=cfg.batchsize, collate_fn=mol_frag_collate, shuffle=True, num_workers = 0)

    valloader = DataLoader(valid_dataset, batch_size=cfg.batchsize, collate_fn=mol_frag_collate, shuffle=False, num_workers = 0)

    testloader = DataLoader(test_dataset, batch_size=cfg.batchsize, collate_fn=mol_frag_collate, shuffle=False, num_workers = 0)

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

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    # criterion = nn.CrossEntropyLoss(reduction="none")
    # optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=0)
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)

    # scheduler = StepLR(optimizer, step_size=30, gamma=0.96)

    best_val = 0
    best_test = None
    best_epoch = 0

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
            pred, feat, losses = model(batch)

            loss_frag_div, loss_frag, loss_tree, loss_mask = losses

            y = batch.y.view(pred.shape).to(torch.float64)
            # Whether y is non-null or not.
            is_valid = y ** 2 > 0
            # Loss matrix
            loss_mat = criterion(pred.double(), (y + 1) / 2)
            # loss matrix after removing null target
            loss_mat = torch.where(
                is_valid, loss_mat,
                torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))

            optimizer.zero_grad()
            loss_task = torch.sum(loss_mat) / torch.sum(is_valid)

            loss = cfg.w_f_d * loss_frag_div + cfg.w_frag * loss_frag + cfg.w_tree * loss_tree + \
                   cfg.w_task * loss_task + cfg.w_mask * loss_mask
            # loss = loss_task

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

        writer.add_scalar(tag='Training Loss', scalar_value=cum_loss / len(trainloader), global_step=epoch)
        # scheduler.step()

        # VAL
        model.eval()
        y_pred = []
        y_true = []
        for step, batch in enumerate(valloader):
            # batch.y[batch.y == -1] = 0
            batch = batch.to(device)
            with torch.no_grad():
                pred, _, losses = model(batch)

            y_true.append(batch['y'].view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        roc_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                is_valid = y_true[:, i]**2 > 0
                # 判断是否存在nan或inf
                if np.isnan(y_pred[is_valid,i]).any():
                    print(f'Epoch {epoch}, Task {i}, nan in y_pred')
                    break
                if np.isinf(y_pred[is_valid,i]).any():
                    print(f'Epoch {epoch}, Task {i}, inf in y_pred')
                s = roc_auc_score((y_true[is_valid,i] + 1)/2, y_pred[is_valid,i], average='macro')
                roc_list.append(s)
            # s = roc_auc_score((y_true[:, i] + 1) / 2, y_pred[:, i])
            # roc_list.append(s)

        val_res = sum(roc_list)/len(roc_list)

        writer.add_scalar(tag='Valid ROC', scalar_value=val_res, global_step=epoch)


        # TEST
        model.eval()
        y_pred = []
        y_true = []

        for step, batch in enumerate(testloader):
            # batch.y[batch.y == -1] = 0
            batch = batch.to(device)
            with torch.no_grad():
                pred, feat, _ = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim = 0).numpy()
        y_pred = torch.cat(y_pred, dim = 0).numpy()


        # roc_list = []
        # for i in range(y_true.shape[1]):
        #     #AUC is only defined when there is at least one positive data.
        #     # if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
        #     #     is_valid = y_true[:,i]**2 > 0
        #     #     roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_pred[is_valid,i]))
        #     roc_list.append(roc_auc_score((y_true[:, i] + 1) / 2, y_pred[:, i]))
        roc_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
                is_valid = y_true[:, i] ** 2 > 0
                roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_pred[is_valid, i], average='macro'))
            else:
                print('{} is invalid'.format(i))

        test_res = sum(roc_list)/len(roc_list)
        writer.add_scalar(tag='Test ROC', scalar_value=test_res, global_step=epoch)
        if val_res > best_val:
            best_val = val_res
            best_test = test_res
            best_model = copy.deepcopy(model)
            best_epoch = epoch

        # print()
        logger.info(
            f"Epoch: {epoch:03d}/{cfg.epoch:03d}, Total Loss: {cum_loss:.4f}, " 
            f"loss_frag_div: {cum_loss_frag_div:.4f}, loss_frag: {cum_loss_frag:.4f}, "
            f"loss_tree: {cum_loss_tree:.4f}, loss_mask:{cum_loss_mask:.4f}, loss_task: {cum_loss_task:.4f},"
            f"Val ROC: {val_res:.4f}, Test ROC: {test_res:.4f}"
        )

    logger.info(f"Best Val ROC: {best_val}, \tBest Test ROC: {best_test}")

    torch.save(best_model.state_dict(), os.path.join(cfg.save_dir, f'{cfg.model}_{best_epoch:03d}.pth'))

    writer.close()

    # feat_e_all = []
    # feat_s_all = []
    # for step, batch in enumerate(testloader):
    #     # batch.y[batch.y == -1] = 0
    #     batch = batch.to(device)
    #     with torch.no_grad():
    #         pred, feat, _ = best_model(batch)
    #         feat_s, feat_e = feat[0], feat[1]
    #         feat_s_all.append(feat_s.detach().cpu())
    #         feat_e_all.append(feat_e.detach().cpu())
    #
    # feat_s_all = torch.cat(feat_s_all, dim=0).numpy()
    # feat_e_all = torch.cat(feat_e_all, dim=0).numpy()
    #
    # np.save(os.path.join(cfg.save_dir, f'feat_s_all.npy'), feat_s_all)
    # np.save(os.path.join(cfg.save_dir, f'feat_e_all.npy'), feat_e_all)