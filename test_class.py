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
    res_path = r'E:\Code\Molecule\MultiMol\result_classification\bbbp\scaffold\all'
    cfg_path = res_path + '\config.yaml'
    model_path = res_path + '\MSE_003.pth'
    # mode = 'train'
    cfg = cfg.from_yaml(cfg_path)

    set_seed(cfg.runseed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print('-' * 60)
    print(f'The setup args are:\n{dict_to_markdown(cfg.to_dict(), max_str_len=120)}')



    # load dataset
    dataset = MoleculeDataset(dataset=cfg.dataset, root=os.path.join(cfg.data_dir, cfg.dataset.lower()))

    # train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list,
    #                                                             null_value=0,
    #                                                             frac_train=0.8,
    #                                                             frac_valid=0.1,
    #                                                             frac_test=0.1)
    smiles_list = pd.read_csv(cfg.data_dir + cfg.dataset + '/processed/smiles.csv', header=None)[0].tolist()

    if cfg.split == 'scaffold':

        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles) = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, return_smiles=True)
        print('split via scaffold')
    elif cfg.split == 'random':
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles) = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=cfg.seed, smiles_list=smiles_list)
        print('randomly split')
    elif cfg.split == 'random_scaffold':
        test_smiles = None
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=cfg.seed)
        print('random scaffold')
    else:
        raise ValueError('Invalid split option.')

    num_tasks = train_dataset[0].y.shape[-1]

    print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Number of tasks: {num_tasks}")

    testloader = DataLoader(train_dataset, batch_size=cfg.batchsize, collate_fn=mol_frag_collate, shuffle=False, num_workers = 0)

    model = MSE(cfg, num_tasks=num_tasks).to(cfg.device)
    state_dict = torch.load(model_path, map_location=device)
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    # 检查不兼容的键
    print(incompatible_keys)

    y_pred = []
    y_true = []

    model.eval()  # 切换到评估模式

    for step, batch in enumerate(testloader):
        # batch.y[batch.y == -1] = 0
        batch = batch.to(device)
        with torch.no_grad():
            pred, feat, _ = model(batch)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_pred[is_valid, i], average='macro'))
        else:
            print('{} is invalid'.format(i))

    test_res = sum(roc_list)/len(roc_list)

    # print()
    print(f"Test ROC: {test_res:.4f}")

    feat_s_raw_all = []
    feat_s_frag_all = []
    feat_s_mam_all = []
    feat_out_all = []
    y_true = []
    y_pred = []
    for step, batch in enumerate(testloader):
        # batch.y[batch.y == -1] = 0
        batch = batch.to(device)
        with torch.no_grad():
            pred, feat, _ = model(batch)
            feat_s_raw, feat_s_frag, feat_s_mam, feat_all = feat[0], feat[1], feat[2], feat[3]

            feat_s_raw_all.append(feat_s_raw.detach().cpu())
            feat_s_frag_all.append(feat_s_frag.detach().cpu())
            feat_s_mam_all.append(feat_s_mam.detach().cpu())
            feat_out_all.append(feat_all.detach().cpu())

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    df = pd.DataFrame(columns=['smiles', 'y_true', 'y_pred'])
    df['smiles'] = pd.Series(train_dataset)
    df['y_true'] = pd.Series(y_true.flatten())
    df['y_pred'] = pd.Series(y_pred.flatten())

    df.to_csv(os.path.join(cfg.save_dir, f'train_results.csv'), index=False)

    feat_s_raw_all = torch.cat(feat_s_raw_all, dim=0).numpy()
    feat_s_frag_all = torch.cat(feat_s_frag_all, dim=0).numpy()
    feat_s_mam_all = torch.cat(feat_s_mam_all, dim=0).numpy()
    feat_out_all = torch.cat(feat_out_all, dim=0).numpy()

    np.save(os.path.join(cfg.save_dir, f'train_feat_s_raw_all.npy'), feat_s_raw_all)
    np.save(os.path.join(cfg.save_dir, f'train_feat_s_frag_all.npy'), feat_s_frag_all)
    np.save(os.path.join(cfg.save_dir, f'train_feat_s_mam_all.npy'), feat_s_mam_all)
    np.save(os.path.join(cfg.save_dir, f'train_feat_out_all.npy'), feat_out_all)