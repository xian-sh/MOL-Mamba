import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import glob
from torch_geometric.loader import DataLoader
# from torch.utils.data import DataLoader
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
from datasets.utils.logger import setup_logger, set_seed 
 
from torch.nn import Module, Linear, ReLU, Sequential, BatchNorm1d 
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool

save_dir = './mlp_train'
model_name = 'mlp'
dataset_name = 'bbbp'
data_dir = r'E:\Code\Molecule\MultiMol\data\chem_dataset'
runseed = 0
split = 'scaffold'
batchsize = 64
seed = 42
lr = 0.0001
epochs = 100
 
logger = setup_logger(f"{model_name}", save_dir, filename=f'train_{model_name}.log')


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz).cuda()
        layers = [
            nn.Dropout(dropout).cuda(),
            nn.Linear(in_hsz, out_hsz).cuda(),
        ]
        self.net = nn.Sequential(*layers).cuda()

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x.float())
        x = self.net(x.float())
        if self.relu:
            x = F.relu(x.float(), inplace=True)
        return x  # (N, L, D)


class MLP(Module):
    def __init__(self, d_in=2, d_h=300, d_o=1):
        super().__init__()

        n_input_proj = 2
        relu_args = [True] * 3
        relu_args[n_input_proj - 1] = False
        self.struct_proj = nn.Sequential(
            *[LinearLayer(d_in, d_h, layer_norm=True,
                          dropout=0.1, relu=relu_args[0]),
              LinearLayer(d_h, d_h, layer_norm=True,
                          dropout=0.1, relu=relu_args[1]),
              LinearLayer(d_h, d_h, layer_norm=True,
                          dropout=0.1, relu=relu_args[2])][:n_input_proj])
        self.mlp_out = Sequential(
            Linear(d_h, d_h),
            ReLU(),
            Linear(d_h, d_o),
        )
        
    def forward(self, data):
        x, batch = data.x.float(), data.batch
        x = self.struct_proj(x)
        feat = global_mean_pool(x, batch)  # (N, d_h)
        x = self.mlp_out(feat)  # (N, 1)
        return x, feat


if __name__ == '__main__':
    set_seed(runseed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
 
    # load dataset
    smiles_list = pd.read_csv(os.path.join(data_dir, dataset_name.lower(), 'processed', 'smiles.csv'), header=None)[
        0].tolist()
    dataset = MoleculeDataset(dataset=dataset_name, root=os.path.join(data_dir, dataset_name.lower()))

    if split == 'scaffold':

        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles) = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, return_smiles=True)
        logger.info('split via scaffold')
    elif split == 'random':
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles) = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed,
            smiles_list=smiles_list)
        logger.info('randomly split')
    elif split == 'random_scaffold':
        test_smiles = None
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed)
        logger.info('random scaffold')
    else:
        raise ValueError('Invalid split option.')

    num_tasks = train_dataset[0].y.shape[-1]

    logger.info(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}, Test samples: {len(test_dataset)}")
    logger.info(f"Number of tasks: {num_tasks}")

    trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

    valloader = DataLoader(valid_dataset, batch_size=batchsize, shuffle=False)

    testloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

    model = MLP(d_in=2, d_h=300, d_o=num_tasks)

    model.to(device)
    best_model = model
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    # criterion = nn.CrossEntropyLoss(reduction="none")
    # optimizer = Adam(model.parameters(), lr=lr, weight_decay=0)
    optimizer = Adam(model.parameters(), lr=lr)

    # scheduler = StepLR(optimizer, step_size=30, gamma=0.96)

    best_val = 0
    best_test = None
    best_epoch = 0

    total_steps = len(trainloader)
    for epoch in range(epochs):
        model.train()
        cum_loss = 0
        for step, batch in enumerate(testloader):

            batch = batch.to(device)
            pred, _ = model(batch)
 
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
            loss = torch.sum(loss_mat) / torch.sum(is_valid)
 
            loss.backward()
            optimizer.step()

            cum_loss += float(loss.cpu().item()) 

        # scheduler.step()

        # VAL
        model.eval()
        y_pred = []
        y_true = []
        for step, batch in enumerate(testloader):
            # batch.y[batch.y == -1] = 0
            batch = batch.to(device)
            with torch.no_grad():
                pred, _ = model(batch)

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
                    logger.info(f'Epoch {epoch}, Task {i}, nan in y_pred')
                    break
                if np.isinf(y_pred[is_valid,i]).any():
                    logger.info(f'Epoch {epoch}, Task {i}, inf in y_pred')
                s = roc_auc_score((y_true[is_valid,i] + 1)/2, y_pred[is_valid,i], average='macro')
                roc_list.append(s)
            # s = roc_auc_score((y_true[:, i] + 1) / 2, y_pred[:, i])
            # roc_list.append(s)

        val_res = sum(roc_list)/len(roc_list)
 
        # TEST
        model.eval()
        y_pred = []
        y_true = []

        for step, batch in enumerate(testloader):
            # batch.y[batch.y == -1] = 0
            batch = batch.to(device)
            with torch.no_grad():
                pred, feat = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim = 0).numpy()
        y_pred = torch.cat(y_pred, dim = 0).numpy()

        roc_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
                is_valid = y_true[:, i] ** 2 > 0
                roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_pred[is_valid, i], average='macro'))
            else:
                logger.info('{} is invalid'.format(i))

        test_res = sum(roc_list)/len(roc_list) 
        if val_res > best_val:
            best_val = val_res
            best_test = test_res
            best_model = copy.deepcopy(model)
            best_epoch = epoch

        # logger.info()
        logger.info(
            f"Epoch: {epoch:03d}/{epochs:03d}, Total Loss: {cum_loss:.4f}, " 
            f"Val ROC: {val_res:.4f}, Test ROC: {test_res:.4f}"
        )

    logger.info(f"Best Val ROC: {best_val}, \tBest Test ROC: {best_test}")

    torch.save(best_model.state_dict(), os.path.join(save_dir, f'{model_name}_{best_epoch:03d}.pth'))
 
    feat_all = []
    y_true = []
    y_pred = []
    for step, batch in enumerate(testloader):
        # batch.y[batch.y == -1] = 0
        batch = batch.to(device)
        with torch.no_grad():
            pred, feat = best_model(batch) 
            feat_all.append(feat.detach().cpu())

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    feat_all = torch.cat(feat_all, dim=0).numpy()
    np.save(os.path.join(save_dir, f'feat_mlp.npy'), feat_all)

    df = pd.DataFrame(columns=['smiles', 'y_true', 'y_pred'])
    df['smiles'] = pd.Series(test_dataset)
    df['y_true'] = pd.Series(y_true.flatten())
    df['y_pred'] = pd.Series(y_pred.flatten())

    df.to_csv(os.path.join(save_dir, f'test_results.csv'), index=False)