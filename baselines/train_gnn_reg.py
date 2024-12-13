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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (MessagePassing, global_add_pool, global_max_pool, global_mean_pool)
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        self.aggr = aggr
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim),
                                 nn.ReLU(),
                                 nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
                          self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()
        self.aggr = aggr
        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)

        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
                          self.edge_embedding2(edge_attr[:, 1])

        norm = self.norm(edge_index[0], x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__(node_dim=0)
        self.aggr = aggr
        self.heads = heads
        self.emb_dim = emb_dim
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, heads * emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)

        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
                          self.edge_embedding2(edge_attr[:, 1])

        x = self.weight_linear(x)
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out += self.bias
        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()
        self.aggr = aggr

        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
                          self.edge_embedding2(edge_attr[:, 1])

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0., gnn_type="gin"):
        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        super(GNN, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.JK = JK

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        # 如果多个维度，相加，如果一个维度，只要一个embedding
        if x.shape[1] == 2:
            x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        elif x.shape[1] > 2:
            pass
        else:
            x = self.x_embedding1(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("not implemented.")
        node_representation[~torch.isfinite(node_representation)] = 0.0
        return node_representation


class GNN_graphpred(nn.Module):
    def __init__(self, args, num_tasks, molecule_model=None):
        super(GNN_graphpred, self).__init__()

        if args.num_layer < 2:
            raise ValueError("# layers must > 1.")

        self.molecule_model = molecule_model
        self.num_layer = args.num_layer
        self.emb_dim = args.emb_dim
        self.num_tasks = num_tasks
        self.JK = args.JK

        # Different kind of graph pooling
        if args.graph_pooling == "sum":
            self.pool = global_add_pool
        elif args.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif args.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim,
                                               self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.mult * self.emb_dim, self.num_tasks)
        return

    def from_pretrained(self, model_file):
        self.molecule_model.load_state_dict(torch.load(model_file))
        return

    def get_graph_representation(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        pred = self.graph_pred_linear(graph_representation)

        return graph_representation, pred

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        output = self.graph_pred_linear(graph_representation)

        return output, graph_representation


model_name = 'graphsage'.lower()
dataset_name = 'esol'.lower()


class Args:
    save_dir = f'E:/Code/Molecule/MultiMol/baselines/{dataset_name}/{model_name}_100'
    model_name = model_name
    dataset_name = dataset_name
    data_dir = r'E:\Code\Molecule\MultiMol\data\chem_dataset'
    runseed = 0
    split = 'random'
    batchsize = 64
    seed = 42
    lr = 0.0001
    epochs = 30
    criterion = 'mse'  # mse
    gnn_type = "gin"  # gin, gcn, gat, graphsage
    num_layer = 6
    emb_dim = 300
    JK = "last"
    graph_pooling = "mean"


if __name__ == '__main__':
    args = Args()
    set_seed(args.runseed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logger = setup_logger(f"{args.model_name}", args.save_dir, filename=f'train_{args.model_name}.log')

    # load dataset

    dataset = MoleculeDataset(dataset=args.dataset_name, root=os.path.join(args.data_dir, args.dataset_name.lower()))
    smiles_list = pd.read_csv(os.path.join(args.data_dir, args.dataset_name.lower(), 'processed', 'smiles.csv'),
                              header=None)[0].tolist()
    if args.split == 'scaffold':
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles) = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, return_smiles=True)
        logger.info('split via scaffold')
    elif args.split == 'random':
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles) = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed,
            smiles_list=smiles_list)
        logger.info('randomly split')
    elif args.split == 'random_scaffold':
        test_smiles = None
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
        logger.info('random scaffold')
    else:
        raise ValueError('Invalid split option.')

    num_tasks = train_dataset[0].y.shape[-1]

    logger.info(
        f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}, Test samples: {len(test_dataset)}")
    logger.info(f"Number of tasks: {num_tasks}")

    trainloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)

    valloader = DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=False)

    testloader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)

    GNN_model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=0., gnn_type=args.gnn_type)
    model = GNN_graphpred(args=args, num_tasks=1, molecule_model=GNN_model)

    model.to(device)
    best_model = model
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    if args.criterion == 'l1':
        criterion = nn.L1Loss()
    elif args.criterion == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError('Invalid criterion.')
    # optimizer = Adam(model.parameters(), lr=lr, weight_decay=0)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=30, gamma=0.96)
    best_val = 10000.
    best_test = None
    best_epoch = 1
    total_steps = len(trainloader)
    for epoch in range(args.epochs):
        model.train()
        cum_loss = 0
        for step, batch in enumerate(trainloader):
            batch = batch.to(device)
            pred, _ = model(batch)
            optimizer.zero_grad()
            # multitask loss
            # pred = pred.view(-1, num_tasks)
            # pred_mean = pred.mean(dim=0)
            y = batch.y.view(-1, num_tasks)
            y_mean = y.mean(dim=1).float()  # b, 1

            loss = criterion(pred, y_mean.view(pred.shape))
            loss.backward()
            optimizer.step()

            cum_loss += float(loss.cpu().item()) 

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
                pred, _ = model(batch)
            y = batch.y.view(-1, num_tasks)
            y_mean = y.mean(dim=1)  # b, 1
            y_true.append(y_mean.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        val_res = torch.sqrt(criterion(y_pred, y_true))

        # TEST
        model.eval()
        y_pred = []
        y_true = []
        for step, batch in enumerate(testloader):
            # batch.y[batch.y == -1] = 0
            batch = batch.to(device)
            with torch.no_grad():
                pred, _ = model(batch)
            y = batch.y.view(-1, num_tasks)
            y_mean = y.mean(dim=1)  # b, 1
            y_true.append(y_mean.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        test_res = torch.sqrt(criterion(y_pred, y_true))

        if val_res < best_val:
            best_val = val_res
            best_test = test_res
            best_model = copy.deepcopy(model)
            best_epoch = epoch

        if args.criterion == 'l1':
            logger.info(
                f"Epoch: {epoch:03d}/{args.epochs:03d}, Total Loss: {cum_loss:.4f}, "
                f"Val MAE: {val_res:.4f}, Test MAE: {test_res:.4f}"
            )
        else:
            logger.info(
            f"Epoch: {epoch:03d}/{args.epochs:03d}, Total Loss: {cum_loss:.4f}, "
            f"Val RMSE: {val_res:.4f}, Test RMSE: {test_res:.4f}"
            )

    if args.criterion == 'l1':
        logger.info(f"Best Val MAE: {best_val.item()}, \tBest Test MAE: {best_test.item()}")
    else:
        logger.info(f"Best Val RMSE: {best_val.item()}, \tBest Test RMSE: {best_test.item()}")

    torch.save(best_model.state_dict(), os.path.join(args.save_dir, f'{args.model_name}_{best_epoch:03d}.pth'))
    
    feat_all = []
    y_true = []
    y_pred = []
    for step, batch in enumerate(testloader):
        # batch.y[batch.y == -1] = 0
        batch = batch.to(device)
        with torch.no_grad():
            pred, feat = best_model(batch)
            feat_all.append(feat.detach().cpu())
            y = batch.y.view(-1, num_tasks)
            y_mean = y.mean(dim=1)
            y_true.append(y_mean.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    feat_all = torch.cat(feat_all, dim=0).numpy()
    np.save(os.path.join(args.save_dir, f'feat_{model_name}.npy'), feat_all)

    df = pd.DataFrame(columns=['smiles', 'y_true', 'y_pred'])
    df['smiles'] = pd.Series(test_dataset)
    df['y_true'] = pd.Series(y_true.flatten())
    df['y_pred'] = pd.Series(y_pred.flatten())

    df.to_csv(os.path.join(args.save_dir, f'test_results.csv'), index=False)
