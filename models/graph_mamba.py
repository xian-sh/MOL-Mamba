import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GINConv, BatchNorm, global_add_pool, radius_graph
from torch_geometric.nn import MLP as PyGMLP
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch.nn import Module, Linear, ReLU, Sequential, BatchNorm1d, Dropout
from einops import rearrange, repeat, einsum
from typing import Union
from dataclasses import dataclass
import math
from torch_geometric.nn import GINEConv, MLP, MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch.nn import Module, Linear, ReLU, Sequential, BatchNorm1d, Dropout
from models.schnet import SchNet, GaussianSmearing
from torch_scatter import scatter, scatter_add
from torch_sparse import SparseTensor
from torch_geometric.utils import degree, sort_edge_index
from models.gnn_model import GNN

class PosEmbedSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images. (To 1D sequences)
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        """
        Args:
            x: torch.tensor, shape (L, d)

        Returns:
            pos_x: Position embeddings of shape (L, num_pos_feats)
        """
        L = x.size(0)  # 输入序列长度
        mask = torch.ones(L, device=x.device)  # 创建 mask，形状为 (L,)
        x_embed = mask.cumsum(0, dtype=torch.float32)  # (L,)

        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[-1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
        pos_x = x_embed[:, None] / dim_t  # (L, num_pos_feats)
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=1).flatten(1)

        return pos_x


class ResidualBlock(nn.Module):
    def __init__(self,
                 gnn,
                 d_model: int = 128,
                 obd: bool = False,
                 obf: bool = False,
                 cutoff: float = 10.0,
                 ):
        super().__init__()

        self.norm = RMSNorm(d_model)
        self.gnn = gnn
        self.cutoff = cutoff
        self.lin = nn.Linear(50, 1)
        self.distance_expansion = GaussianSmearing(0.0, 10, 50)

        self.degree_emb = PosEmbedSine(num_pos_feats=d_model, normalize=True)
        self.frag_emb = PosEmbedSine(num_pos_feats=d_model, normalize=True)
        self.order_by_degree = obd
        self.order_by_frag = obf

        self.mixer = MambaBlock(d_model)

    def forward(self, x, pos, map, batch):
        x1 = self.norm(x)  # [n, d]
        x2 = self.gnn(x1, pos, batch)

        # 从gnn获取距离矩阵和邻接矩阵
        # gnn_edge, gnn_dis = self.gnn.get_edge_index(), self.gnn.get_edge_weight()
        gnn_edge = radius_graph(pos, r=self.cutoff, batch=batch)
        if gnn_edge.numel() == 0:
            raise ValueError("gnn_edge is empty. Please check the input data.")

        if self.order_by_frag:
            order_frag = torch.stack([batch, map], 1).T
            sort_frag, x2 = sort_edge_index(order_frag, edge_attr=x2)
            _, map = sort_edge_index(order_frag, edge_attr=map)
            _, pos = sort_edge_index(order_frag, edge_attr=pos)
            _, x1 = sort_edge_index(order_frag, edge_attr=x1)

            # print('x20', x2.shape)
            x2 = x2 + self.frag_emb(x2)

        # sort based on the degree of nodes, batch
        if self.order_by_degree:
            deg = degree(gnn_edge[0], x2.shape[0]).to(torch.long)
            order_deg = torch.stack([batch, deg], 1).T
            sort_deg, x2 = sort_edge_index(order_deg, edge_attr=x2)
            _, map = sort_edge_index(order_deg, edge_attr=map)
            _, pos = sort_edge_index(order_deg, edge_attr=pos)
            _, x1 = sort_edge_index(order_deg, edge_attr=x1)

            x2 = x2 + self.degree_emb(x2)

        # 遍历每个分子，根绝node_batch
        unique_batches = torch.unique(batch)

        x3 = []
        for i, batch_idx in enumerate(unique_batches):
            # 获取当前 batch 的节点索引
            mask = (batch == batch_idx)
            x_t = x2[mask]
            p_t = pos[mask]

            e_t = radius_graph(p_t, r=self.cutoff)
            if e_t.numel() == 0:
                # print(p_t)
                # raise ValueError("e_t is empty. Please check the input data.")
                dis_dense = torch.ones(10, 10, device=x.device)
            else:
                row, col = e_t
                edge_weight = (p_t[row] - p_t[col]).norm(dim=-1)
                d_t = self.distance_expansion(edge_weight)
                d_t = self.lin(d_t).squeeze(1)
                num_nodes = max(e_t[0].max(), e_t[1].max()) + 1  # x2.size(0)
                dis_sparse = SparseTensor(row=e_t[0], col=e_t[1], value=d_t, sparse_sizes=(num_nodes, num_nodes))
                dis_dense = dis_sparse.to_dense()

            x3.append(self.mixer(x_t, dis_dense))

        x3 = torch.cat(x3, dim=0)

        output = x3 + x1
        return output, pos, map


class Graph_Mamba(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 n_layer: int = 4,
                 sch_layer: int = 4,
                 dim_in: int = 2,
                 cutoff: float = 5.0,
                 ):
        super().__init__()

        # self.gin = GNN(num_layer=4, emb_dim=d_model)
        self.gnn = SchNet(hidden_channels=d_model,
                          num_filters=d_model,
                          num_interactions=sch_layer,
                          num_gaussians=50,
                          cutoff=cutoff,
                          readout='mean',
                          )
        self.encode = nn.Linear(dim_in, d_model)
        if n_layer == 1:
            pass
        else:
            self.encoder_layers = nn.ModuleList()
            self.encoder_layers.append(ResidualBlock(self.gnn, d_model, obd=True, obf=True, cutoff=cutoff))
            for _ in range(n_layer-1):
                self.encoder_layers.append(ResidualBlock(self.gnn, d_model, obd=False, obf=False, cutoff=cutoff))

        # readout
        self.encoder_norm = RMSNorm(d_model)
        self.decode = nn.Linear(d_model, d_model)

        self.frag_pred = nn.Linear(d_model, 3200)
        self.tree_pred = nn.Linear(d_model, 3000)
        self.pool = global_mean_pool

    def forward(self, data):
        x, map, pos, edge_index, batch = data.x.float(), data.map, data.pos, data.edge_index, data.node_batch
        x = self.encode(x)  # [n, d]
        # x = self.gin(x, edge_index, data.edge_attr)
        for layer in self.encoder_layers:
            x, pos, map = layer(x, pos, map, batch)
        x = self.encoder_norm(x)
        x = self.decode(x)

        x[~torch.isfinite(x)] = 0.0
        frag_emb = scatter(x, map, dim=0, reduce="mean")
        pred_frag = self.pool(self.frag_pred(x), batch)
        pred_tree = self.pool(self.tree_pred(x), batch)
        # pred_tree[~torch.isfinite(pred_tree)] = 0.0
        # pred_frag[~torch.isfinite(pred_frag)] = 0.0

        return x, frag_emb, pred_frag, pred_tree


class MambaBlock(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 bias: bool = False,
                 conv_bias: bool = True,
                 d_conv: int = 4,
                 dt_rank: Union[int, str] = 'auto',
                 d_state: int = 2,
                 ):
        super().__init__()

        self.in_proj = nn.Linear(d_model, d_model * 2, bias=bias)
        self.d_model = d_model

        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_model,
            padding=d_conv - 1,
        )

        if dt_rank == 'auto':
            dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        self.x_proj = nn.Linear(d_model, dt_rank + d_state * 2, bias=False)

        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_model)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, dis_dense):
        x = x.unsqueeze(0)
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.d_model, self.d_model], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)

        y = self.ssm(x, dis_dense)

        y = y * F.silu(res)

        output = self.out_proj(y)

        output = output.squeeze(0)
        output[~torch.isfinite(output)] = 0.0

        return output

    def ssm(self, x, dis_dense):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D, dis_dense)

        return y

    def selective_scan(self, u, delta, A, B, C, D, dis_dense):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        # This is the new version of Selective Scan Algorithm named as "Graph Selective Scan"
        # In Graph Selective Scan, we use the Feed-Forward graph information from KFGN, and incorporate the Feed-Forward information with "delta"
        temp_adj = dis_dense
        if temp_adj.size(0) >= d_in:
            temp_adj = temp_adj[:d_in, :d_in]
        temp_adj_padded = torch.ones(d_in, d_in, device=temp_adj.device)
        # print(temp_adj_padded.shape)
        temp_adj_padded[:temp_adj.size(0), :temp_adj.size(1)] = temp_adj

        delta_p = torch.matmul(delta, temp_adj_padded)

        # The fused param delta_p will participate in the following upgrading of deltaA and deltaB_u
        deltaA = torch.exp(einsum(delta_p, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta_p, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


if __name__ == '__main__':
    from torch_geometric.data import Data
    from datasets.loader_pretrain import MoleculePretrainDataset, mol_frag_collate
    from torch.utils.data import DataLoader

    # Instantiate the model
    model = Graph_Mamba(d_model=128, n_layer=4, dim_in=2).cuda()

    root = r'E:\Code\Molecule\MultiMol\data\chem_dataset\bbbp'
    data_file_path = r'E:\Code\Molecule\MultiMol\data\chem_dataset\bbbp\raw\BBBP.csv'
    vocab_file_path = r'E:\Code\Molecule\MultiMol\datasets\vocab.txt'

    dataset = MoleculePretrainDataset(root=root,
                                      smiles_column='smiles',
                                      data_file_path=data_file_path,
                                      vocab_file_path=vocab_file_path)

    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=mol_frag_collate)

    for data in loader:
        x, frag_emb, pred_frag, pred_tree = model(data.cuda())
        print(x.shape, frag_emb.shape, pred_frag.shape, pred_tree.shape)
        break