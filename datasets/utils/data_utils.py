import torch
from torch_geometric.data import Data


def mol_frag_collate(data_list):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects.
    The assignment vector :obj:`batch` is created on the fly."""

    batch = Data()
    # keys follow node
    node_sum_keys = ["edge_index"]
    # keys follow frag
    frag_sum_keys = ["frag_edge_index", "map"]
    # no sum keys
    # no_sum_keys = ["edge_attr",
    #                "x",
    #                "y",
    #                "pos",
    #                "map",
    #                "frag",
    #                "descriptors",
    #                "frag_unique"]
    no_sum_keys = ["edge_attr",
                   "x",
                   "pos",
                   "descriptors",
                   "frag",
                   "frag_unique"
                   ]

    for key in node_sum_keys + frag_sum_keys + no_sum_keys:
        batch[key] = []

    batch.y = []

    batch.node_batch_size = []
    batch.node_batch = []

    batch.frag_batch_size = []
    batch.frag_batch = []

    cumsum_node = 0
    i_node = 0

    cumsum_frag = 0
    i_frag = 0

    for data in data_list:
        num_nodes = data.x.shape[0]

        num_frags = data.frag.shape[0]

        batch.node_batch_size.append(num_nodes)

        batch.frag_batch_size.append(num_frags)

        batch.node_batch.append(torch.full((num_nodes,), i_node, dtype=torch.long))

        batch.frag_batch.append(torch.full((num_frags,), i_frag, dtype=torch.long))

        batch.y.append(data.y)

        for key in node_sum_keys:
            item = data[key]
            item = item + cumsum_node
            batch[key].append(item)

        for key in frag_sum_keys:
            item = data[key]
            item = item + cumsum_frag
            batch[key].append(item)

        for key in no_sum_keys:
            item = data[key]
            batch[key].append(item)

        cumsum_node += num_nodes
        i_node += 1

        cumsum_frag += num_frags
        i_frag += 1

    batch.x = torch.cat(batch.x, dim=0)
    batch.pos = torch.cat(batch.pos, dim=0)
    batch.edge_index = torch.cat(batch.edge_index, dim=-1)
    batch.edge_attr = torch.cat(batch.edge_attr, dim=0)
    batch.descriptors = torch.cat(batch.descriptors, dim=0)
    batch.frag = torch.cat(batch.frag, dim=0)
    batch.frag_edge_index = torch.cat(batch.frag_edge_index, dim=-1)
    batch.frag_unique = torch.cat(batch.frag_unique, dim=0)
    batch.map = torch.cat(batch.map, dim=-1)
    # for key in keys:
    #     batch[key] = torch.cat(
    #         batch[key], dim=batch.cat_dim(key))
    batch.node_batch = torch.cat(batch.node_batch, dim=-1)
    batch.node_batch_size = torch.tensor(batch.node_batch_size)
    batch.frag_batch = torch.cat(batch.frag_batch, dim=-1)
    batch.frag_batch_size = torch.tensor(batch.frag_batch_size)

    batch.y = torch.stack(batch.y)

    batch.tree = torch.LongTensor([data.tree for data in data_list])

    return batch.contiguous()