import torch
import torch.nn as nn
import math
from typing import Union
from torch.nn import Module, Linear, ReLU, Sequential, BatchNorm1d, Dropout
from dataclasses import dataclass
from torch.functional import F
from torch_scatter import scatter
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing
from models.graph_mamba import Graph_Mamba, PosEmbedSine
from models.gnn_model import GINE, GNN
from models.mamba import Mamba
from models.loss_info_nce import InfoNCE
from torch_geometric.utils import add_self_loops


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(5, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(3, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        if edge_attr != None:
            # add features corresponding to self-loop edges.
            self_loop_attr = torch.zeros(x.size(0), 2)
            self_loop_attr[:, 0] = 4  # bond type for self-loop edge
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

            edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        else:
            edge_embeddings = torch.zeros((edge_index[0].shape[1], x.shape[-1])).to(x.device)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GIN(torch.nn.Module):
    """

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, drop_ratio=0, atom=False):
        super(GIN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if atom:
            self.x_embedding1 = torch.nn.Embedding(120, emb_dim)
            self.x_embedding2 = torch.nn.Embedding(3, emb_dim)

            torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
            torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        else:
            self.embedding = nn.Embedding(908, emb_dim)

        self.atom = atom

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        if self.atom:
            x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        else:
            x = self.embedding(x[:, 0])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            # print(h.shape)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        node_representation = h_list[-1]

        return node_representation


class FragEmbedding(torch.nn.Module):
    def __init__(self,
                 emb_dim=300,
                 num_gnn_layers=5,
                 dropout=0.0):
        super(FragEmbedding, self).__init__()

        self.gnn = GIN(num_layer=num_gnn_layers,
                       emb_dim=emb_dim,
                       drop_ratio=dropout,
                       atom=False)

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x, edge_index, edge_attr=None):
        x = self.gnn(x, edge_index, edge_attr)
        return self.proj(x)


class PositionEmbeddingSine(nn.Module):
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
            x: torch.tensor, (batch_size, L, d)
            mask: torch.tensor, (batch_size, L), with 1 as valid

        Returns:

        """
        # assert mask is not None
        mask = torch.ones(1, x.size(1), device=x.device)  # (bsz, l)
        x_embed = mask.cumsum(1, dtype=torch.float32)  # (bsz, L)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t  # (bsz, L, num_pos_feats)
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)  # (bsz, L, num_pos_feats*2)
        # import ipdb; ipdb.set_trace()
        return pos_x  # .permute(0, 2, 1)  # (bsz, num_pos_feats*2, L)


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


# class MLP(Module):
#     def __init__(self, d_in, d_h, d_o):
#         super().__init__()
#
#         self.layers = Sequential(
#             BatchNorm1d(d_in),
#             Linear(d_in, d_h),
#             ReLU(),
#             Linear(d_h, d_o),
#         )
#
#     def forward(self, x):
#         return self.layers(x)


# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size_1)
#         self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
#         self.fc3 = nn.Linear(hidden_size_2, output_size)
#         self.activation = nn.ReLU()
#
#     def forward(self, x):
#         x = self.activation(self.fc1(x))
#         x = self.activation(self.fc2(x))
#         x = self.fc3(x)
#         return x


class MFuser(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 d_hidden: int = 1024,
                 heads: int = 1,
                 attn_dropout: float = 0.0,
                 dropout: float = 0.1,
                 ):
        super().__init__()

        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

        self._reset_parameters()

        self.attn = torch.nn.MultiheadAttention(d_model, heads, dropout=attn_dropout, batch_first=True,)
        # self.mamba = Mamba(d_model=d_model, n_layer=4, dim_in=d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos):
        src = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos = pos.permute(1, 0, 2)  # (L, batch_size, d)

        src2 = self.norm1(src)
        src2 = src2 + pos
        src2_1 = self.attn(src2, src2, src2)[0]
        # src2_2 = self.mamba(src2)
        src2 = src2 + self.dropout1(src2_1)

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        src = self.norm3(src)
        src = src.transpose(0, 1)
        return src


def criterion(y, pred):
    fun = nn.BCEWithLogitsLoss(reduction="none")
    is_valid = y ** 2 > 0
    # Loss matrix
    loss_mat = fun(pred.double(), (y + 1) / 2)
    # loss matrix after removing null target
    loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

    loss = torch.sum(loss_mat) / torch.sum(is_valid)
    return loss


class MSE(nn.Module):
    def __init__(self, args, num_tasks=1):
        super().__init__()

        self.args = args
        self.num_tasks = num_tasks

        n_input_proj = 2
        relu_args = [True] * 3
        relu_args[n_input_proj - 1] = False
        self.struct_proj = nn.Sequential(
            *[LinearLayer(2, args.d_model, layer_norm=True,
                          dropout=0.5, relu=relu_args[0]),
              LinearLayer(args.d_model, args.d_model, layer_norm=True,
                          dropout=0.5, relu=relu_args[1]),
              LinearLayer(args.d_model, args.d_model, layer_norm=True,
                          dropout=0.5, relu=relu_args[2])][:n_input_proj])

        if self.args.use_gnn:
            self.gnn_encoder = GNN(num_layer=args.gnn_layer, emb_dim=args.emb_dim).to(args.device)

            self.gnn_proj = nn.Sequential(
                *[LinearLayer(args.d_model, args.d_model, layer_norm=True,
                              dropout=0.5, relu=relu_args[0]),
                  LinearLayer(args.d_model, args.d_model, layer_norm=True,
                              dropout=0.5, relu=relu_args[1]),
                  LinearLayer(args.d_model, args.d_model, layer_norm=True,
                              dropout=0.5, relu=relu_args[2])][:n_input_proj])

        if args.use_frag_gnn:
            self.frag_encoder = FragEmbedding(emb_dim=args.d_model, num_gnn_layers=args.frag_layer, dropout=0.0).to(args.device)
            #GNN(num_layer=args.frag_layer, emb_dim=args.emb_dim).to(args.device)
                # FragEmbedding(emb_dim=args.d_model, num_gnn_layers=args.frag_layer, dropout=0.0).to(args.device)

        self.elec_proj = nn.Sequential(
            *[LinearLayer(1, args.d_model, layer_norm=True,
                          dropout=0.5, relu=relu_args[0]),
              LinearLayer(args.d_model, args.d_model, layer_norm=True,
                          dropout=0.5, relu=relu_args[1]),
              LinearLayer(args.d_model, args.d_model, layer_norm=True,
                          dropout=0.5, relu=relu_args[2])][:n_input_proj])

        if self.args.use_graph_mamba:
            self.graph_mamba_encoder = Graph_Mamba(d_model=args.d_model, n_layer=args.n_layer,
                                               sch_layer=args.sch_layer, dim_in=2, cutoff=args.cutoff).to(args.device)
        else:
            self.frag_pred = nn.Linear(args.d_model, 3200)
            self.tree_pred = nn.Linear(args.d_model, 3000)

        # self.elec_encoder = MLP(args.d_in, args.d_h, args.d_o).to(args.device)

        if args.use_fusion:
            self.mfuser = MFuser(d_model=args.d_model, heads=1, attn_dropout=0.0).to(args.device)
            self.struct_pos = PositionEmbeddingSine(num_pos_feats=args.d_model)
            self.elec_pos = PositionEmbeddingSine(num_pos_feats=args.d_model)

            self.mlp_e = Sequential(
                Linear(args.d_model, args.d_model // 2),
                ReLU(),
                Dropout(p=args.drop),
                Linear(args.d_model // 2, args.d_model // 4),
                ReLU(),
                Dropout(p=args.drop),
                Linear(args.d_model // 4, 1),
            )

        else:
            self.mlp_gnn = Sequential(
                Linear(args.d_model, args.d_model // 2),
                ReLU(),
                Dropout(p=args.drop),
                Linear(args.d_model // 2, args.d_model // 4),
                ReLU(),
                Dropout(p=args.drop),
                Linear(args.d_model // 4, num_tasks),
            )

            self.mlp_mam = Sequential(
                Linear(args.d_model, args.d_model // 2),
                ReLU(),
                Dropout(p=args.drop),
                Linear(args.d_model // 2, args.d_model // 4),
                ReLU(),
                Dropout(p=args.drop),
                Linear(args.d_model // 4, num_tasks),
            )

            self.mlp_e = Sequential(
                Linear(args.d_model, args.d_model // 2),
                ReLU(),
                Dropout(p=args.drop),
                Linear(args.d_model // 2, args.d_model // 4),
                ReLU(),
                Dropout(p=args.drop),
                Linear(args.d_model // 4, num_tasks),
            )

        # base settings
        self.softmax = nn.Softmax(dim=1)
        self.frag_div_loss = nn.CrossEntropyLoss()
        # self.frag_div_loss = InfoNCE()
        # self.frag_div_loss = nn.KLDivLoss(reduction="batchmean")
        self.frag_loss = nn.L1Loss()
        self.tree_loss = nn.CrossEntropyLoss()
        self.mask_loss = nn.MSELoss()  # nn.L1Loss()

        self.mlp = Sequential(
            Linear(args.d_model, args.d_model // 2),
            ReLU(),
            Dropout(p=args.drop),
            Linear(args.d_model // 2, args.d_model // 4),
            ReLU(),
            Dropout(p=args.drop),
            Linear(args.d_model // 4, num_tasks),
        )

        # if args.task_type == 'classification':
        #     # self.task_loss = criterion
        #     self.task_loss = nn.BCEWithLogitsLoss(reduction="none")
        # elif args.task_type == 'regression':
        #     self.task_loss = nn.MSELoss()  # nn.L1Loss()
        # else:
        #     raise ValueError("Invalid task type")

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        x, edge_index, edge_attr, map = data.x, data.edge_index, data.edge_attr, data.map

        h = self.struct_proj(x)

        # initialization
        if self.args.use_gnn:
            x_g = self.gnn_encoder(x, edge_index, edge_attr)
            x_g = self.gnn_proj(x_g) + h
        else:
            x_g = h

        feat_s_raw = global_mean_pool(h, data.node_batch).unsqueeze(1)

        x_g_frag = scatter(x_g, map, dim=0, reduce="mean")
        x_g_frag[~torch.isfinite(x_g_frag)] = 0.0
        x_g_f_d = self.softmax(x_g_frag)

        # fragment embedding
        if self.args.use_frag_gnn:
            x_f_frag = self.frag_encoder(data.frag, data.frag_edge_index, None)
            x_f_frag[~torch.isfinite(x_g_frag)] = 0.0
            x_f_f_d = self.softmax(x_g_frag)
            loss_frag_div1 = self.frag_div_loss(x_f_f_d, x_g_f_d)
        else:
            x_f_frag = None
            loss_frag_div1 = torch.zeros(1).to(x.device)

        feat_s_frag = global_mean_pool(x_f_frag, data.frag_batch).unsqueeze(1)

        # graph mamba
        if self.args.use_graph_mamba:
            x_m, x_m_frag, pred_frag, pred_tree = self.graph_mamba_encoder(data)
            x_m_frag[~torch.isfinite(x_m_frag)] = 0.0
            x_m_f_d = self.softmax(x_m_frag)

            # structure losses
            loss_frag_div2 = self.frag_div_loss(x_m_f_d, x_g_f_d)  # input, target
        else:
            x_m = None
            pred_frag = global_mean_pool(self.frag_pred(x_g), data.node_batch)
            pred_tree = global_mean_pool(self.tree_pred(x_g), data.node_batch)

            loss_frag_div2 = torch.zeros(1).to(x.device)

        feat_s_mam = global_mean_pool(x_m, data.node_batch).unsqueeze(1)

        loss_frag_div = loss_frag_div1 + loss_frag_div2
        loss_frag = self.frag_loss(pred_frag, data.frag_unique.reshape(pred_frag.shape))
        loss_tree = self.tree_loss(pred_tree, data.tree)

        # 电子
        # 遍历每个分子，根绝node_batch
        unique_batches = torch.unique(data.node_batch)
        bs = max(unique_batches) + 1
        input_e = data.descriptors.reshape(-1, 1)  # 209*bs, 1
        l_e = torch.div(input_e.size(0), bs).int()  # 209
        input_e = input_e.reshape(bs, l_e, input_e.size(1))  # b,l,1

        feat_e_org = input_e.clone().transpose(1, 2)  # b, 1, 209

        num_mask = int(l_e * self.args.mask_ratio)
        mask_idx = torch.randperm(l_e)[:num_mask]

        if self.args.mask_e:
            masked_e = input_e.clone()
            masked_e[torch.arange(bs)[:, None], mask_idx] = 0
            # 归一化
            # masked_e = masked_e / (masked_e.norm(dim=1, keepdim=True) + 1e-10)
            x_e = self.elec_proj(masked_e)  # b, l, d

        else:
            x_e = self.elec_proj(input_e)

        if self.args.use_fusion:
            if self.args.use_graph_mamba:
                x_s = x_g + x_m  # l, d
            else:
                x_s = x_g

            feat_s_out = []
            feat_e_out = []
            feat_all = []

            results = []
            results_e = []
            for i, batch_idx in enumerate(unique_batches):
                # 获取当前 batch 的节点索引
                mask = (data.node_batch == batch_idx)
                s = x_s[mask].unsqueeze(0)  # 1, n, d
                e = x_e[i].unsqueeze(0)  # 1, l, d

                p_s = self.struct_pos(s)  # 1, n, d
                p_e = self.elec_pos(e)

                src = torch.cat([s, e], dim=1)  # 1, l+n, d
                pos = torch.cat([p_s, p_e], dim=1)

                memory = self.mfuser(src, pos)  # 1, l+n, d
                out = self.mlp(memory.squeeze(0))
                out = torch.mean(out, dim=0, keepdim=True)  # 1, num_tasks
                out[~torch.isfinite(out)] = 0.0

                results.append(out)

                s_mem = memory[:, e.shape[1]:]  # (batch_size, L_txt, d)
                e_mem = memory[:, :e.shape[1]]  # (batch_size, L_vid, d)   1 128 256

                feat_s_out.append(torch.mean(s_mem, dim=1, keepdim=True))
                feat_e_out.append(torch.mean(e_mem, dim=1, keepdim=True))
                feat_all.append(torch.mean(memory, dim=1, keepdim=True))

                out_e = self.mlp_e(e_mem)
                # out_e[~torch.isfinite(out_e)] = 0.0

                results_e.append(out_e)

            # print(data.y.shape)
            results = torch.stack(results, dim=0).reshape(data.y.size(0), data.y.size(1))
            results_e = torch.stack(results_e, dim=0).squeeze(1)

            feat_s_out = torch.cat(feat_s_out, dim=0)
            feat_e_out = torch.cat(feat_e_out, dim=0)
            feat_all = torch.cat(feat_all, dim=0)

            # 对mask值归一化
            r_e = results_e[torch.arange(bs)[:, None], mask_idx]
            i_e = input_e[torch.arange(bs)[:, None], mask_idx]
            r_e[~torch.isfinite(r_e)] = 0.0
            r_e[~torch.isfinite(i_e)] = 0.0
            r_e_s = self.softmax(r_e)
            i_e_s = self.softmax(i_e)
            # 归一化

            loss_mask = self.mask_loss(r_e_s, i_e_s)

        else:
            feat_s_out = global_mean_pool(x_g + x_m, data.node_batch).unsqueeze(1)  # b, 1, d
            feat_e_out = torch.mean(x_e.reshape(bs, l_e, -1), dim=1, keepdim=True)  # b, 209, d
            feat_all = None

            # print(feat_s.shape, feat_e.shape)
            out_gnn = global_mean_pool(self.mlp_gnn(x_g), data.node_batch).reshape(-1, self.num_tasks)
            out_mam = global_mean_pool(self.mlp_mam(x_m), data.node_batch).reshape(-1, self.num_tasks)
            out_e = self.mlp_e(x_e.squeeze(0))  # b, l, num_tasks
            out_e[~torch.isfinite(out_e)] = 0.0
            out_e = torch.mean(out_e, dim=1, keepdim=True).squeeze(1)

            results = out_gnn + out_mam + out_e
            loss_mask = torch.zeros(1).to(x.device)

        return results, [feat_s_raw, feat_s_frag, feat_s_mam, feat_all], [loss_frag_div, loss_frag, loss_tree, loss_mask]