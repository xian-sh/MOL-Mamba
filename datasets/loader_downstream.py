# Borrowed from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py

import os
import torch
import pickle
import collections
import math
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch_geometric.utils import to_networkx
from networkx import weisfeiler_lehman_graph_hash
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
from datasets.fragment.mol_bpe import Tokenizer
from . import mol_to_graph_data_obj_pos, create_standardized_mol_id

organic_major_ish = {'[C]', '[O]', '[N]', '[F]', '[Cl]', '[Br]', '[I]', '[S]', '[P]', '[B]', '[H]'}

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from .utils.logger import setup_logger

logger = setup_logger('MSE.Data', './', filename="data_log.txt")
logger.info('-'*40)
logger.info("Running the loader_downstream.py")

# 定义要计算的分子描述符列表
descriptor_names = [desc[0] for desc in Descriptors._descList]

# 创建分子描述符计算器
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)




class MoleculeDataset(InMemoryDataset):
    def __init__(self, dataset, root, vocab_file_path=r'E:\Code\Molecule\MultiMol\datasets\vocab.txt'):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param dataset: name of the dataset. Currently only implemented for zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast

        :param root: directory of the dataset, containing a raw and processed dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        """
        self.dataset = dataset
        self.root = root
        self.vocab_file_path = vocab_file_path
        if self.vocab_file_path:
            self.tokenizer = Tokenizer(vocab_file_path)
            self.vocab_dict = {smiles: i for i, smiles in enumerate(self.tokenizer.vocab_dict.keys())}
        self.root = root

        super(MoleculeDataset, self).__init__(self.root, transform=None, pre_transform=None)

        self.transform, self.pre_transform, self.pre_filter = None, None, None

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []

        if self.dataset == 'zinc_standard_agent':
            input_path = self.raw_paths[0]
            input_df = pd.read_csv(input_path, sep=',', compression='gzip',
                                   dtype='str')
            smiles_list = list(input_df['smiles'])
            zinc_id_list = list(input_df['zinc_id'])
            for i in tqdm(range(len(smiles_list))):
                s = smiles_list[i]
                # each example contains a single species
                try:
                    rdkit_mol = AllChem.MolFromSmiles(s)
                    if rdkit_mol != None:  # ignore invalid mol objects
                        # # convert aromatic bonds to double bonds
                        # Chem.SanitizeMol(rdkit_mol,
                        #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                        data = mol_to_graph_data_obj_pos(rdkit_mol)
                        # manually add mol id
                        id = int(zinc_id_list[i].split('ZINC')[1].lstrip('0'))
                        data.id = torch.tensor(
                            [id])  # id here is zinc id value, stripped of
                        # leading zeros
                        data_list.append(data)
                        data_smiles_list.append(smiles_list[i])
                except:
                    continue

        elif self.dataset == 'chembl_filtered':
            ### get downstream test molecules.
            from utils.splitters import scaffold_split

            ###
            downstream_dir = [
                'dataset/bace',
                'dataset/bbbp',
                'dataset/clintox',
                'dataset/esol',
                'dataset/freesolv',
                'dataset/hiv',
                'dataset/lipophilicity',
                'dataset/muv',
                # 'dataset/pcba/processed/smiles.csv',
                'dataset/sider',
                'dataset/tox21',
                'dataset/toxcast'
            ]

            downstream_inchi_set = set()
            for d_path in downstream_dir:
                logger.info(d_path)
                dataset_name = d_path.split('/')[1]
                downstream_dataset = MoleculeDataset(d_path, dataset=dataset_name)
                downstream_smiles = pd.read_csv(os.path.join(d_path,
                                                             'processed', 'smiles.csv'),
                                                header=None)[0].tolist()

                assert len(downstream_dataset) == len(downstream_smiles)

                _, _, _, (train_smiles, valid_smiles, test_smiles) = scaffold_split(downstream_dataset,
                                                                                    downstream_smiles, task_idx=None,
                                                                                    null_value=0,
                                                                                    frac_train=0.8, frac_valid=0.1,
                                                                                    frac_test=0.1,
                                                                                    return_smiles=True)

                ### remove both test and validation molecules
                remove_smiles = test_smiles + valid_smiles

                downstream_inchis = []
                for smiles in remove_smiles:
                    species_list = smiles.split('.')
                    for s in species_list:  # record inchi for all species, not just
                        # largest (by default in create_standardized_mol_id if input has
                        # multiple species)
                        inchi = create_standardized_mol_id(s)
                        downstream_inchis.append(inchi)
                downstream_inchi_set.update(downstream_inchis)

            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_chembl_with_labels_dataset(os.path.join(self.root, 'raw'))

            logger.info('processing')
            for i in range(len(rdkit_mol_objs)):
                logger.info(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    mw = Descriptors.MolWt(rdkit_mol)
                    if 50 <= mw <= 900:
                        inchi = create_standardized_mol_id(smiles_list[i])
                        if inchi != None and inchi not in downstream_inchi_set:
                            data = mol_to_graph_data_obj_pos(rdkit_mol)
                            # manually add mol id
                            data.id = torch.tensor(
                                [i])  # id here is the index of the mol in
                            # the dataset
                            data.y = torch.tensor(labels[i, :])
                            # fold information
                            if i in folds[0]:
                                data.fold = torch.tensor([0])
                            elif i in folds[1]:
                                data.fold = torch.tensor([1])
                            else:
                                data.fold = torch.tensor([2])
                            data_list.append(data)
                            data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = _load_tox21_dataset(self.raw_paths[0])
            for i in tqdm(range(len(smiles_list))):
                rdkit_mol = rdkit_mol_objs[i]
                smiles = smiles_list[i]


                data = mol_to_graph_data_obj_pos(rdkit_mol)

                # descriptors-----------------------------------------
                descriptors = calculator.CalcDescriptors(rdkit_mol)
                data.descriptors = torch.Tensor(descriptors)
                data.id = torch.tensor([i])  # id here is the index of the mol in the dataset
                data.y = torch.tensor(labels[i, :])

                # tree-------------------------------------------------
                try:
                    tree = self.tokenizer(smiles)
                except:
                    logger.info(f"Line {i}, Unable to process SMILES: {smiles}")
                    continue

                # Manually consructing the fragment graph
                map = [0] * data.num_nodes
                frag = [[0] for _ in range(len(tree.nodes))]
                frag_edge_index = [[], []]

                try:
                    for node_i in tree.nodes:
                        node = tree.get_node(node_i)
                        # for atom in node, set map
                        for atom_i in node.atom_mapping.keys():
                            map[atom_i] = node_i
                            # extend frag
                            frag[node_i][0] = self.vocab_dict[node.smiles]
                    for src, dst in tree.edges:
                        # extend edge index
                        frag_edge_index[0].extend([src, dst])
                        frag_edge_index[1].extend([dst, src])
                except KeyError as e:
                    logger.info(f"Line {i}, Error in matching subgraphs {e}")
                    continue

                unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
                frag_unique = torch.zeros(3200).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

                data.map = torch.LongTensor(map)
                data.frag = torch.LongTensor(frag)
                data.frag_edge_index = torch.LongTensor(frag_edge_index)
                data.frag_unique = frag_unique

                data_list.append(data)

                data_smiles_list.append(smiles_list[i])

            tree_dict = {}
            hash_str_list = []
            for data in data_list:
                tree = Data()
                tree.x = data.frag
                tree.edge_index = data.frag_edge_index
                nx_graph = to_networkx(tree, to_undirected=True)
                hash_str = weisfeiler_lehman_graph_hash(nx_graph)
                if hash_str not in tree_dict:
                    tree_dict[hash_str] = len(tree_dict)
                hash_str_list.append(hash_str)

            tree = []
            for hash_str in hash_str_list:
                tree.append(tree_dict[hash_str])

            for i, data in enumerate(data_list):
                data.tree = tree[i]

        elif self.dataset == 'hiv':
            smiles_list, rdkit_mol_objs, labels = _load_hiv_dataset(self.raw_paths[0])
            for i in tqdm(range(len(smiles_list))):
                rdkit_mol = rdkit_mol_objs[i]
                smiles = smiles_list[i]

                data = mol_to_graph_data_obj_pos(rdkit_mol)

                # descriptors-----------------------------------------
                descriptors = calculator.CalcDescriptors(rdkit_mol)
                data.descriptors = torch.Tensor(descriptors)
                data.id = torch.tensor([i])  # id here is the index of the mol in the dataset
                data.y = torch.tensor([labels[i]])

                # tree-------------------------------------------------
                try:
                    tree = self.tokenizer(smiles)
                except:
                    logger.info(f"Line {i}, Unable to process SMILES: {smiles}")
                    continue

                # Manually consructing the fragment graph
                map = [0] * data.num_nodes
                frag = [[0] for _ in range(len(tree.nodes))]
                frag_edge_index = [[], []]

                try:
                    for node_i in tree.nodes:
                        node = tree.get_node(node_i)
                        # for atom in node, set map
                        for atom_i in node.atom_mapping.keys():
                            map[atom_i] = node_i
                            # extend frag
                            frag[node_i][0] = self.vocab_dict[node.smiles]
                    for src, dst in tree.edges:
                        # extend edge index
                        frag_edge_index[0].extend([src, dst])
                        frag_edge_index[1].extend([dst, src])
                except KeyError as e:
                    logger.info(f"Line {i}, Error in matching subgraphs {e}")
                    continue

                unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
                frag_unique = torch.zeros(3200).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

                data.map = torch.LongTensor(map)
                data.frag = torch.LongTensor(frag)
                data.frag_edge_index = torch.LongTensor(frag_edge_index)
                data.frag_unique = frag_unique

                data_list.append(data)

                data_smiles_list.append(smiles_list[i])

            tree_dict = {}
            hash_str_list = []
            for data in data_list:
                tree = Data()
                tree.x = data.frag
                tree.edge_index = data.frag_edge_index
                nx_graph = to_networkx(tree, to_undirected=True)
                hash_str = weisfeiler_lehman_graph_hash(nx_graph)
                if hash_str not in tree_dict:
                    tree_dict[hash_str] = len(tree_dict)
                hash_str_list.append(hash_str)

            tree = []
            for hash_str in hash_str_list:
                tree.append(tree_dict[hash_str])

            for i, data in enumerate(data_list):
                data.tree = tree[i]

        elif self.dataset == 'bace':
            smiles_list, rdkit_mol_objs, folds, labels = _load_bace_dataset(self.raw_paths[0])
            for i in tqdm(range(len(smiles_list))):
                rdkit_mol = rdkit_mol_objs[i]
                smiles = smiles_list[i]

                data = mol_to_graph_data_obj_pos(rdkit_mol)

                # descriptors-----------------------------------------
                descriptors = calculator.CalcDescriptors(rdkit_mol)
                data.descriptors = torch.Tensor(descriptors)
                data.id = torch.tensor([i])  # id here is the index of the mol in the dataset
                data.y = torch.tensor([labels[i]])
                data.fold = torch.tensor([folds[i]])

                # tree-------------------------------------------------
                try:
                    tree = self.tokenizer(smiles)
                except:
                    logger.info(f"Line {i}, Unable to process SMILES: {smiles}")
                    continue

                # Manually consructing the fragment graph
                map = [0] * data.num_nodes
                frag = [[0] for _ in range(len(tree.nodes))]
                frag_edge_index = [[], []]

                try:
                    for node_i in tree.nodes:
                        node = tree.get_node(node_i)
                        # for atom in node, set map
                        for atom_i in node.atom_mapping.keys():
                            map[atom_i] = node_i
                            # extend frag
                            frag[node_i][0] = self.vocab_dict[node.smiles]
                    for src, dst in tree.edges:
                        # extend edge index
                        frag_edge_index[0].extend([src, dst])
                        frag_edge_index[1].extend([dst, src])
                except KeyError as e:
                    logger.info(f"Line {i}, Error in matching subgraphs {e}")
                    continue

                unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
                frag_unique = torch.zeros(3200).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

                data.map = torch.LongTensor(map)
                data.frag = torch.LongTensor(frag)
                data.frag_edge_index = torch.LongTensor(frag_edge_index)
                data.frag_unique = frag_unique

                data_list.append(data)

                data_smiles_list.append(smiles_list[i])

            tree_dict = {}
            hash_str_list = []
            for data in data_list:
                tree = Data()
                tree.x = data.frag
                tree.edge_index = data.frag_edge_index
                nx_graph = to_networkx(tree, to_undirected=True)
                hash_str = weisfeiler_lehman_graph_hash(nx_graph)
                if hash_str not in tree_dict:
                    tree_dict[hash_str] = len(tree_dict)
                hash_str_list.append(hash_str)

            tree = []
            for hash_str in hash_str_list:
                tree.append(tree_dict[hash_str])

            for i, data in enumerate(data_list):
                data.tree = tree[i]

        elif self.dataset == 'bbbp':
            smiles_list, rdkit_mol_objs, labels = _load_bbbp_dataset(self.raw_paths[0])
            for i in tqdm(range(len(smiles_list))):
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol == None:
                    continue
                smiles = smiles_list[i]

                data = mol_to_graph_data_obj_pos(rdkit_mol)

                # descriptors-----------------------------------------
                descriptors = calculator.CalcDescriptors(rdkit_mol)
                data.descriptors = torch.Tensor(descriptors)
                data.id = torch.tensor([i])  # id here is the index of the mol in the dataset
                data.y = torch.tensor([labels[i]])

                # tree-------------------------------------------------
                try:
                    tree = self.tokenizer(smiles)
                except:
                    logger.info(f"Line {i}, Unable to process SMILES: {smiles}")
                    continue

                # Manually consructing the fragment graph
                map = [0] * data.num_nodes
                frag = [[0] for _ in range(len(tree.nodes))]
                frag_edge_index = [[], []]

                try:
                    for node_i in tree.nodes:
                        node = tree.get_node(node_i)
                        # for atom in node, set map
                        for atom_i in node.atom_mapping.keys():
                            map[atom_i] = node_i
                            # extend frag
                            frag[node_i][0] = self.vocab_dict[node.smiles]
                    for src, dst in tree.edges:
                        # extend edge index
                        frag_edge_index[0].extend([src, dst])
                        frag_edge_index[1].extend([dst, src])
                except KeyError as e:
                    logger.info(f"Line {i}, Error in matching subgraphs {e}")
                    continue

                unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
                frag_unique = torch.zeros(3200).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

                data.map = torch.LongTensor(map)
                data.frag = torch.LongTensor(frag)
                data.frag_edge_index = torch.LongTensor(frag_edge_index)
                data.frag_unique = frag_unique

                data_list.append(data)

                data_smiles_list.append(smiles_list[i])

            tree_dict = {}
            hash_str_list = []
            for data in data_list:
                tree = Data()
                tree.x = data.frag
                tree.edge_index = data.frag_edge_index
                nx_graph = to_networkx(tree, to_undirected=True)
                hash_str = weisfeiler_lehman_graph_hash(nx_graph)
                if hash_str not in tree_dict:
                    tree_dict[hash_str] = len(tree_dict)
                hash_str_list.append(hash_str)

            tree = []
            for hash_str in hash_str_list:
                tree.append(tree_dict[hash_str])

            for i, data in enumerate(data_list):
                data.tree = tree[i]

        elif self.dataset == 'clintox':
            smiles_list, rdkit_mol_objs, labels = _load_clintox_dataset(self.raw_paths[0])
            for i in tqdm(range(len(smiles_list))):
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol == None:
                    continue
                smiles = smiles_list[i]

                data = mol_to_graph_data_obj_pos(rdkit_mol)

                # descriptors-----------------------------------------
                descriptors = calculator.CalcDescriptors(rdkit_mol)
                data.descriptors = torch.Tensor(descriptors)
                data.id = torch.tensor([i])  # id here is the index of the mol in the dataset
                data.y = torch.tensor(labels[i, :])

                # tree-------------------------------------------------
                try:
                    tree = self.tokenizer(smiles)
                except:
                    logger.info(f"Line {i}, Unable to process SMILES: {smiles}")
                    continue

                # Manually consructing the fragment graph
                map = [0] * data.num_nodes
                frag = [[0] for _ in range(len(tree.nodes))]
                frag_edge_index = [[], []]

                try:
                    for node_i in tree.nodes:
                        node = tree.get_node(node_i)
                        # for atom in node, set map
                        for atom_i in node.atom_mapping.keys():
                            map[atom_i] = node_i
                            # extend frag
                            frag[node_i][0] = self.vocab_dict[node.smiles]
                    for src, dst in tree.edges:
                        # extend edge index
                        frag_edge_index[0].extend([src, dst])
                        frag_edge_index[1].extend([dst, src])
                except KeyError as e:
                    logger.info(f"Line {i}, Error in matching subgraphs {e}")
                    continue

                unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
                frag_unique = torch.zeros(3200).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

                data.map = torch.LongTensor(map)
                data.frag = torch.LongTensor(frag)
                data.frag_edge_index = torch.LongTensor(frag_edge_index)
                data.frag_unique = frag_unique

                data_list.append(data)

                data_smiles_list.append(smiles_list[i])

            tree_dict = {}
            hash_str_list = []
            for data in data_list:
                tree = Data()
                tree.x = data.frag
                tree.edge_index = data.frag_edge_index
                nx_graph = to_networkx(tree, to_undirected=True)
                hash_str = weisfeiler_lehman_graph_hash(nx_graph)
                if hash_str not in tree_dict:
                    tree_dict[hash_str] = len(tree_dict)
                hash_str_list.append(hash_str)

            tree = []
            for hash_str in hash_str_list:
                tree.append(tree_dict[hash_str])

            for i, data in enumerate(data_list):
                data.tree = tree[i]

        elif self.dataset == 'esol':
            smiles_list, rdkit_mol_objs, labels = _load_esol_dataset(self.raw_paths[0])
            for i in tqdm(range(len(smiles_list))):
                rdkit_mol = rdkit_mol_objs[i]
                smiles = smiles_list[i]

                data = mol_to_graph_data_obj_pos(rdkit_mol)

                # descriptors-----------------------------------------
                descriptors = calculator.CalcDescriptors(rdkit_mol)
                data.descriptors = torch.Tensor(descriptors)
                data.id = torch.tensor([i])  # id here is the index of the mol in the dataset
                data.y = torch.tensor([labels[i]])

                # tree-------------------------------------------------
                try:
                    tree = self.tokenizer(smiles)
                except:
                    logger.info(f"Line {i}, Unable to process SMILES: {smiles}")
                    continue

                # Manually consructing the fragment graph
                map = [0] * data.num_nodes
                frag = [[0] for _ in range(len(tree.nodes))]
                frag_edge_index = [[], []]

                try:
                    for node_i in tree.nodes:
                        node = tree.get_node(node_i)
                        # for atom in node, set map
                        for atom_i in node.atom_mapping.keys():
                            map[atom_i] = node_i
                            # extend frag
                            frag[node_i][0] = self.vocab_dict[node.smiles]
                    for src, dst in tree.edges:
                        # extend edge index
                        frag_edge_index[0].extend([src, dst])
                        frag_edge_index[1].extend([dst, src])
                except KeyError as e:
                    logger.info(f"Line {i}, Error in matching subgraphs {e}")
                    continue

                unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
                frag_unique = torch.zeros(3200).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

                data.map = torch.LongTensor(map)
                data.frag = torch.LongTensor(frag)
                data.frag_edge_index = torch.LongTensor(frag_edge_index)
                data.frag_unique = frag_unique

                data_list.append(data)

                data_smiles_list.append(smiles_list[i])

            tree_dict = {}
            hash_str_list = []
            for data in data_list:
                tree = Data()
                tree.x = data.frag
                tree.edge_index = data.frag_edge_index
                nx_graph = to_networkx(tree, to_undirected=True)
                hash_str = weisfeiler_lehman_graph_hash(nx_graph)
                if hash_str not in tree_dict:
                    tree_dict[hash_str] = len(tree_dict)
                hash_str_list.append(hash_str)

            tree = []
            for hash_str in hash_str_list:
                tree.append(tree_dict[hash_str])

            for i, data in enumerate(data_list):
                data.tree = tree[i]

        elif self.dataset == 'freesolv':
            smiles_list, rdkit_mol_objs, labels = _load_freesolv_dataset(self.raw_paths[0])
            for i in tqdm(range(len(smiles_list))):
                rdkit_mol = rdkit_mol_objs[i]
                smiles = smiles_list[i]

                data = mol_to_graph_data_obj_pos(rdkit_mol)

                # descriptors-----------------------------------------
                descriptors = calculator.CalcDescriptors(rdkit_mol)
                data.descriptors = torch.Tensor(descriptors)
                data.id = torch.tensor([i])  # id here is the index of the mol in the dataset
                data.y = torch.tensor([labels[i]])

                # tree-------------------------------------------------
                try:
                    tree = self.tokenizer(smiles)
                except:
                    logger.info(f"Line {i}, Unable to process SMILES: {smiles}")
                    continue

                # Manually consructing the fragment graph
                map = [0] * data.num_nodes
                frag = [[0] for _ in range(len(tree.nodes))]
                frag_edge_index = [[], []]

                try:
                    for node_i in tree.nodes:
                        node = tree.get_node(node_i)
                        # for atom in node, set map
                        for atom_i in node.atom_mapping.keys():
                            map[atom_i] = node_i
                            # extend frag
                            frag[node_i][0] = self.vocab_dict[node.smiles]
                    for src, dst in tree.edges:
                        # extend edge index
                        frag_edge_index[0].extend([src, dst])
                        frag_edge_index[1].extend([dst, src])
                except KeyError as e:
                    logger.info(f"Line {i}, Error in matching subgraphs {e}")
                    continue

                unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
                frag_unique = torch.zeros(3200).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

                data.map = torch.LongTensor(map)
                data.frag = torch.LongTensor(frag)
                data.frag_edge_index = torch.LongTensor(frag_edge_index)
                data.frag_unique = frag_unique

                data_list.append(data)

                data_smiles_list.append(smiles_list[i])

            tree_dict = {}
            hash_str_list = []
            for data in data_list:
                tree = Data()
                tree.x = data.frag
                tree.edge_index = data.frag_edge_index
                nx_graph = to_networkx(tree, to_undirected=True)
                hash_str = weisfeiler_lehman_graph_hash(nx_graph)
                if hash_str not in tree_dict:
                    tree_dict[hash_str] = len(tree_dict)
                hash_str_list.append(hash_str)

            tree = []
            for hash_str in hash_str_list:
                tree.append(tree_dict[hash_str])

            for i, data in enumerate(data_list):
                data.tree = tree[i]

        elif self.dataset == 'lipophilicity':
            smiles_list, rdkit_mol_objs, labels = _load_lipophilicity_dataset(self.raw_paths[0])
            for i in tqdm(range(len(smiles_list))):
                rdkit_mol = rdkit_mol_objs[i]
                smiles = smiles_list[i]

                data = mol_to_graph_data_obj_pos(rdkit_mol)

                # descriptors-----------------------------------------
                descriptors = calculator.CalcDescriptors(rdkit_mol)
                data.descriptors = torch.Tensor(descriptors)
                data.id = torch.tensor([i])  # id here is the index of the mol in the dataset
                data.y = torch.tensor([labels[i]])

                # tree-------------------------------------------------
                try:
                    tree = self.tokenizer(smiles)
                except:
                    logger.info(f"Line {i}, Unable to process SMILES: {smiles}")
                    continue

                # Manually consructing the fragment graph
                map = [0] * data.num_nodes
                frag = [[0] for _ in range(len(tree.nodes))]
                frag_edge_index = [[], []]

                try:
                    for node_i in tree.nodes:
                        node = tree.get_node(node_i)
                        # for atom in node, set map
                        for atom_i in node.atom_mapping.keys():
                            map[atom_i] = node_i
                            # extend frag
                            frag[node_i][0] = self.vocab_dict[node.smiles]
                    for src, dst in tree.edges:
                        # extend edge index
                        frag_edge_index[0].extend([src, dst])
                        frag_edge_index[1].extend([dst, src])
                except KeyError as e:
                    logger.info(f"Line {i}, Error in matching subgraphs {e}")
                    continue

                unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
                frag_unique = torch.zeros(3200).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

                data.map = torch.LongTensor(map)
                data.frag = torch.LongTensor(frag)
                data.frag_edge_index = torch.LongTensor(frag_edge_index)
                data.frag_unique = frag_unique

                data_list.append(data)

                data_smiles_list.append(smiles_list[i])

            tree_dict = {}
            hash_str_list = []
            for data in data_list:
                tree = Data()
                tree.x = data.frag
                tree.edge_index = data.frag_edge_index
                nx_graph = to_networkx(tree, to_undirected=True)
                hash_str = weisfeiler_lehman_graph_hash(nx_graph)
                if hash_str not in tree_dict:
                    tree_dict[hash_str] = len(tree_dict)
                hash_str_list.append(hash_str)

            tree = []
            for hash_str in hash_str_list:
                tree.append(tree_dict[hash_str])

            for i, data in enumerate(data_list):
                data.tree = tree[i]

        elif self.dataset == 'muv':
            smiles_list, rdkit_mol_objs, labels = _load_muv_dataset(self.raw_paths[0])
            for i in tqdm(range(len(smiles_list))):
                rdkit_mol = rdkit_mol_objs[i]
                smiles = smiles_list[i]

                data = mol_to_graph_data_obj_pos(rdkit_mol)

                # descriptors-----------------------------------------
                descriptors = calculator.CalcDescriptors(rdkit_mol)
                data.descriptors = torch.Tensor(descriptors)
                data.id = torch.tensor([i])  # id here is the index of the mol in the dataset
                data.y = torch.tensor(labels[i, :])

                # tree-------------------------------------------------
                try:
                    tree = self.tokenizer(smiles)
                except:
                    logger.info(f"Line {i}, Unable to process SMILES: {smiles}")
                    continue

                # Manually consructing the fragment graph
                map = [0] * data.num_nodes
                frag = [[0] for _ in range(len(tree.nodes))]
                frag_edge_index = [[], []]

                try:
                    for node_i in tree.nodes:
                        node = tree.get_node(node_i)
                        # for atom in node, set map
                        for atom_i in node.atom_mapping.keys():
                            map[atom_i] = node_i
                            # extend frag
                            frag[node_i][0] = self.vocab_dict[node.smiles]
                    for src, dst in tree.edges:
                        # extend edge index
                        frag_edge_index[0].extend([src, dst])
                        frag_edge_index[1].extend([dst, src])
                except KeyError as e:
                    logger.info(f"Line {i}, Error in matching subgraphs {e}")
                    continue

                unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
                frag_unique = torch.zeros(3200).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

                data.map = torch.LongTensor(map)
                data.frag = torch.LongTensor(frag)
                data.frag_edge_index = torch.LongTensor(frag_edge_index)
                data.frag_unique = frag_unique

                data_list.append(data)

                data_smiles_list.append(smiles_list[i])

            tree_dict = {}
            hash_str_list = []
            for data in data_list:
                tree = Data()
                tree.x = data.frag
                tree.edge_index = data.frag_edge_index
                nx_graph = to_networkx(tree, to_undirected=True)
                hash_str = weisfeiler_lehman_graph_hash(nx_graph)
                if hash_str not in tree_dict:
                    tree_dict[hash_str] = len(tree_dict)
                hash_str_list.append(hash_str)

            tree = []
            for hash_str in hash_str_list:
                tree.append(tree_dict[hash_str])

            for i, data in enumerate(data_list):
                data.tree = tree[i]

        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = _load_sider_dataset(self.raw_paths[0])
            for i in tqdm(range(len(smiles_list))):
                rdkit_mol = rdkit_mol_objs[i]
                smiles = smiles_list[i]

                data = mol_to_graph_data_obj_pos(rdkit_mol)

                # descriptors-----------------------------------------
                descriptors = calculator.CalcDescriptors(rdkit_mol)
                data.descriptors = torch.Tensor(descriptors)
                data.id = torch.tensor([i])  # id here is the index of the mol in the dataset
                data.y = torch.tensor(labels[i, :])

                # tree-------------------------------------------------
                try:
                    tree = self.tokenizer(smiles)
                except:
                    logger.info(f"Line {i}, Unable to process SMILES: {smiles}")
                    continue

                # Manually consructing the fragment graph
                map = [0] * data.num_nodes
                frag = [[0] for _ in range(len(tree.nodes))]
                frag_edge_index = [[], []]

                try:
                    for node_i in tree.nodes:
                        node = tree.get_node(node_i)
                        # for atom in node, set map
                        for atom_i in node.atom_mapping.keys():
                            map[atom_i] = node_i
                            # extend frag
                            frag[node_i][0] = self.vocab_dict[node.smiles]
                    for src, dst in tree.edges:
                        # extend edge index
                        frag_edge_index[0].extend([src, dst])
                        frag_edge_index[1].extend([dst, src])
                except KeyError as e:
                    logger.info(f"Line {i}, Error in matching subgraphs {e}")
                    continue

                unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
                frag_unique = torch.zeros(3200).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

                data.map = torch.LongTensor(map)
                data.frag = torch.LongTensor(frag)
                data.frag_edge_index = torch.LongTensor(frag_edge_index)
                data.frag_unique = frag_unique

                data_list.append(data)

                data_smiles_list.append(smiles_list[i])

            tree_dict = {}
            hash_str_list = []
            for data in data_list:
                tree = Data()
                tree.x = data.frag
                tree.edge_index = data.frag_edge_index
                nx_graph = to_networkx(tree, to_undirected=True)
                hash_str = weisfeiler_lehman_graph_hash(nx_graph)
                if hash_str not in tree_dict:
                    tree_dict[hash_str] = len(tree_dict)
                hash_str_list.append(hash_str)

            tree = []
            for hash_str in hash_str_list:
                tree.append(tree_dict[hash_str])

            for i, data in enumerate(data_list):
                data.tree = tree[i]

        elif self.dataset == 'toxcast':
            smiles_list, rdkit_mol_objs, labels = _load_toxcast_dataset(self.raw_paths[0])
            for i in tqdm(range(len(smiles_list))):
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol == None:
                    continue
                smiles = smiles_list[i]

                data = mol_to_graph_data_obj_pos(rdkit_mol)

                # descriptors-----------------------------------------
                descriptors = calculator.CalcDescriptors(rdkit_mol)
                data.descriptors = torch.Tensor(descriptors)
                data.id = torch.tensor([i])  # id here is the index of the mol in the dataset
                data.y = torch.tensor(labels[i, :])

                # tree-------------------------------------------------
                try:
                    tree = self.tokenizer(smiles)
                except:
                    logger.info(f"Line {i}, Unable to process SMILES: {smiles}")
                    continue

                # Manually consructing the fragment graph
                map = [0] * data.num_nodes
                frag = [[0] for _ in range(len(tree.nodes))]
                frag_edge_index = [[], []]

                try:
                    for node_i in tree.nodes:
                        node = tree.get_node(node_i)
                        # for atom in node, set map
                        for atom_i in node.atom_mapping.keys():
                            map[atom_i] = node_i
                            # extend frag
                            frag[node_i][0] = self.vocab_dict[node.smiles]
                    for src, dst in tree.edges:
                        # extend edge index
                        frag_edge_index[0].extend([src, dst])
                        frag_edge_index[1].extend([dst, src])
                except KeyError as e:
                    logger.info(f"Line {i}, Error in matching subgraphs {e}")
                    continue

                unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
                frag_unique = torch.zeros(3200).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

                data.map = torch.LongTensor(map)
                data.frag = torch.LongTensor(frag)
                data.frag_edge_index = torch.LongTensor(frag_edge_index)
                data.frag_unique = frag_unique

                data_list.append(data)

                data_smiles_list.append(smiles_list[i])

            tree_dict = {}
            hash_str_list = []
            for data in data_list:
                tree = Data()
                tree.x = data.frag
                tree.edge_index = data.frag_edge_index
                nx_graph = to_networkx(tree, to_undirected=True)
                hash_str = weisfeiler_lehman_graph_hash(nx_graph)
                if hash_str not in tree_dict:
                    tree_dict[hash_str] = len(tree_dict)
                hash_str_list.append(hash_str)

            tree = []
            for hash_str in hash_str_list:
                tree.append(tree_dict[hash_str])

            for i, data in enumerate(data_list):
                data.tree = tree[i]

        elif self.dataset == 'qm9':
            smiles_list, rdkit_mol_objs, labels = _load_qm9_dataset(self.raw_paths[0])
            for i in tqdm(range(len(smiles_list))):
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol == None:
                    continue
                smiles = smiles_list[i]

                data = mol_to_graph_data_obj_pos(rdkit_mol)

                # descriptors-----------------------------------------
                descriptors = calculator.CalcDescriptors(rdkit_mol)
                data.descriptors = torch.Tensor(descriptors)
                data.id = torch.tensor([i])  # id here is the index of the mol in the dataset
                data.y = torch.tensor(labels[i, :])

                # tree-------------------------------------------------
                try:
                    tree = self.tokenizer(smiles)
                except:
                    logger.info(f"Line {i}, Unable to process SMILES: {smiles}")
                    continue

                # Manually consructing the fragment graph
                map = [0] * data.num_nodes
                frag = [[0] for _ in range(len(tree.nodes))]
                frag_edge_index = [[], []]

                try:
                    for node_i in tree.nodes:
                        node = tree.get_node(node_i)
                        # for atom in node, set map
                        for atom_i in node.atom_mapping.keys():
                            map[atom_i] = node_i
                            # extend frag
                            frag[node_i][0] = self.vocab_dict[node.smiles]
                    for src, dst in tree.edges:
                        # extend edge index
                        frag_edge_index[0].extend([src, dst])
                        frag_edge_index[1].extend([dst, src])
                except KeyError as e:
                    logger.info(f"Line {i}, Error in matching subgraphs {e}")
                    continue

                unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
                frag_unique = torch.zeros(3200).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

                data.map = torch.LongTensor(map)
                data.frag = torch.LongTensor(frag)
                data.frag_edge_index = torch.LongTensor(frag_edge_index)
                data.frag_unique = frag_unique

                data_list.append(data)

                data_smiles_list.append(smiles_list[i])

            tree_dict = {}
            hash_str_list = []
            for data in data_list:
                tree = Data()
                tree.x = data.frag
                tree.edge_index = data.frag_edge_index
                nx_graph = to_networkx(tree, to_undirected=True)
                hash_str = weisfeiler_lehman_graph_hash(nx_graph)
                if hash_str not in tree_dict:
                    tree_dict[hash_str] = len(tree_dict)
                hash_str_list.append(hash_str)

            tree = []
            for hash_str in hash_str_list:
                tree.append(tree_dict[hash_str])

            for i, data in enumerate(data_list):
                data.tree = tree[i]
        # elif self.dataset == 'ptc_mr':
        #     input_path = self.raw_paths[0]
        #     input_df = pd.read_csv(input_path, sep=',', header=None, names=['id', 'label', 'smiles'])
        #     smiles_list = input_df['smiles']
        #     labels = input_df['label'].values
        #     for i in range(len(smiles_list)):
        #         logger.info(i)
        #         s = smiles_list[i]
        #         rdkit_mol = AllChem.MolFromSmiles(s)
        #         if rdkit_mol != None:  # ignore invalid mol objects
        #             # # convert aromatic bonds to double bonds
        #             # Chem.SanitizeMol(rdkit_mol,
        #             #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #             data = mol_to_graph_data_obj_pos(rdkit_mol)
        #             # manually add mol id
        #             data.id = torch.tensor(
        #                 [i])
        #             data.y = torch.tensor([labels[i]])
        #             data_list.append(data)
        #             data_smiles_list.append(smiles_list[i])

        # elif self.dataset == 'mutag':
        #     smiles_path = os.path.join(self.root, 'raw', 'mutag_188_data.can')
        #     # smiles_path = 'dataset/mutag/raw/mutag_188_data.can'
        #     labels_path = os.path.join(self.root, 'raw', 'mutag_188_target.txt')
        #     # labels_path = 'dataset/mutag/raw/mutag_188_target.txt'
        #     smiles_list = pd.read_csv(smiles_path, sep=' ', header=None)[0]
        #     labels = pd.read_csv(labels_path, header=None)[0].values
        #     for i in range(len(smiles_list)):
        #         logger.info(i)
        #         s = smiles_list[i]
        #         rdkit_mol = AllChem.MolFromSmiles(s)
        #         if rdkit_mol != None:  # ignore invalid mol objects
        #             # # convert aromatic bonds to double bonds
        #             # Chem.SanitizeMol(rdkit_mol,
        #             #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        #             data = mol_to_graph_data_obj_pos(rdkit_mol)
        #             # manually add mol id
        #             data.id = torch.tensor(
        #                 [i])
        #             data.y = torch.tensor([labels[i]])
        #             data_list.append(data)
        #             data_smiles_list.append(smiles_list[i])

        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# NB: only properly tested when dataset_1 is chembl_with_labels and dataset_2
# is pcba_pretrain
def merge_dataset_objs(dataset_1, dataset_2):
    """
    Naively merge 2 molecule dataset objects, and ignore identities of
    molecules. Assumes both datasets have multiple y labels, and will pad
    accordingly. ie if dataset_1 has obj_1 with y dim 1310 and dataset_2 has
    obj_2 with y dim 128, then the resulting obj_1 and obj_2 will have dim
    1438, where obj_1 have the last 128 cols with 0, and obj_2 have
    the first 1310 cols with 0.
    :return: pytorch geometric dataset obj, with the x, edge_attr, edge_index,
    new y attributes only
    """
    d_1_y_dim = dataset_1[0].y.size()[0]
    d_2_y_dim = dataset_2[0].y.size()[0]

    data_list = []
    # keep only x, edge_attr, edge_index, padded_y then append
    for d in dataset_1:
        old_y = d.y
        new_y = torch.cat([old_y, torch.zeros(d_2_y_dim, dtype=torch.long)])
        data_list.append(Data(x=d.x, edge_index=d.edge_index,
                              edge_attr=d.edge_attr, y=new_y))

    for d in dataset_2:
        old_y = d.y
        new_y = torch.cat([torch.zeros(d_1_y_dim, dtype=torch.long), old_y.long()])
        data_list.append(Data(x=d.x, edge_index=d.edge_index,
                              edge_attr=d.edge_attr, y=new_y))

    # create 'empty' dataset obj. Just randomly pick a dataset and root path
    # that has already been processed
    new_dataset = MoleculeDataset(root='dataset/chembl_with_labels',
                                  dataset='chembl_with_labels', empty=True)
    # collate manually
    new_dataset.data, new_dataset.slices = new_dataset.collate(data_list)

    return new_dataset


def create_circular_fingerprint(mol, radius, size, chirality):
    """

    :param mol:
    :param radius:
    :param size:
    :param chirality:
    :return: np array of morgan fingerprint
    """
    fp = GetMorganFingerprintAsBitVect(mol, radius,
                                       nBits=size, useChirality=chirality)
    return np.array(fp)


class MoleculeFingerprintDataset(data.Dataset):
    def __init__(self, root, dataset, radius, size, chirality=True):
        """
        Create dataset object containing list of dicts, where each dict
        contains the circular fingerprint of the molecule, label, id,
        and possibly precomputed fold information
        :param root: directory of the dataset, containing a raw and
        processed_fp dir. The raw dir should contain the file containing the
        smiles, and the processed_fp dir can either be empty or a
        previously processed file
        :param dataset: name of dataset. Currently only implemented for
        tox21, hiv, chembl_with_labels
        :param radius: radius of the circular fingerprints
        :param size: size of the folded fingerprint vector
        :param chirality: if True, fingerprint includes chirality information
        """
        self.dataset = dataset
        self.root = root
        self.radius = radius
        self.size = size
        self.chirality = chirality

        self._load()

    def _process(self):
        data_smiles_list = []
        data_list = []
        if self.dataset == 'chembl_with_labels':
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_chembl_with_labels_dataset(os.path.join(self.root, 'raw'))
            logger.info('processing')
            for i in range(len(rdkit_mol_objs)):
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    fp_arr = create_circular_fingerprint(rdkit_mol,
                                                         self.radius,
                                                         self.size, self.chirality)
                    fp_arr = torch.tensor(fp_arr)
                    # manually add mol id
                    id = torch.tensor([i])  # id here is the index of the mol in
                    # the dataset
                    y = torch.tensor(labels[i, :])
                    # fold information
                    if i in folds[0]:
                        fold = torch.tensor([0])
                    elif i in folds[1]:
                        fold = torch.tensor([1])
                    else:
                        fold = torch.tensor([2])
                    data_list.append({'fp_arr': fp_arr, 'id': id, 'y': y,
                                      'fold': fold})
                    data_smiles_list.append(smiles_list[i])
        elif self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = _load_tox21_dataset(os.path.join(self.root, 'raw/tox21.csv'))
            logger.info('processing')
            for i in tqdm(range(len(smiles_list))):
                rdkit_mol = rdkit_mol_objs[i]
                ## convert aromatic bonds to double bonds
                fp_arr = create_circular_fingerprint(rdkit_mol,
                                                     self.radius,
                                                     self.size,
                                                     self.chirality)
                fp_arr = torch.tensor(fp_arr)

                # manually add mol id
                id = torch.tensor([i])  # id here is the index of the mol in
                # the dataset
                y = torch.tensor(labels[i, :])
                data_list.append({'fp_arr': fp_arr, 'id': id, 'y': y})
                data_smiles_list.append(smiles_list[i])
        elif self.dataset == 'hiv':
            smiles_list, rdkit_mol_objs, labels =  _load_hiv_dataset(os.path.join(self.root, 'raw/HIV.csv'))
            logger.info('processing')
            for i in tqdm(range(len(smiles_list))):
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                fp_arr = create_circular_fingerprint(rdkit_mol,
                                                     self.radius,
                                                     self.size,
                                                     self.chirality)
                fp_arr = torch.tensor(fp_arr)

                # manually add mol id
                id = torch.tensor([i])  # id here is the index of the mol in
                # the dataset
                y = torch.tensor([labels[i]])
                data_list.append({'fp_arr': fp_arr, 'id': id, 'y': y})
                data_smiles_list.append(smiles_list[i])
        else:
            raise ValueError('Invalid dataset name')

        # save processed data objects and smiles
        processed_dir = os.path.join(self.root, 'processed_fp')
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(processed_dir, 'smiles.csv'),
                                  index=False,
                                  header=False)
        with open(os.path.join(processed_dir,
                               'fingerprint_data_processed.pkl'),
                  'wb') as f:
            pickle.dump(data_list, f)

    def _load(self):
        processed_dir = os.path.join(self.root, 'processed_fp')
        # check if saved file exist. If so, then load from save
        file_name_list = os.listdir(processed_dir)
        if 'fingerprint_data_processed.pkl' in file_name_list:
            with open(os.path.join(processed_dir,
                                   'fingerprint_data_processed.pkl'),
                      'rb') as f:
                self.data_list = pickle.load(f)
        # if no saved file exist, then perform processing steps, save then
        # reload
        else:
            self._process()
            self._load()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        ## if iterable class is passed, return dataset objection
        if hasattr(index, "__iter__"):
            dataset = MoleculeFingerprintDataset(self.root, self.dataset, self.radius, self.size,
                                                 chirality=self.chirality)
            dataset.data_list = [self.data_list[i] for i in index]
            return dataset
        else:
            return self.data_list[index]


def _load_tox21_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_hiv_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['HIV_active']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_bace_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array
    containing indices for each of the 3 folds, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['mol']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['Class']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    folds = input_df['Model']
    folds = folds.replace('Train', 0)  # 0 -> train
    folds = folds.replace('Valid', 1)  # 1 -> valid
    folds = folds.replace('Test', 2)  # 2 -> test
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)
    return smiles_list, rdkit_mol_objs_list, folds.values, labels.values


def _load_bbbp_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df['p_np']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
        labels.values


def _load_clintox_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    tasks = ['FDA_APPROVED', 'CT_TOX']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
        labels.values


# input_path = 'dataset/clintox/raw/clintox.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_clintox_dataset(input_path)

def _load_esol_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    # NB: some examples have multiple species
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured log solubility in mols per litre']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


# input_path = 'dataset/esol/raw/delaney-processed.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_esol_dataset(input_path)

def _load_freesolv_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['expt']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_lipophilicity_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['exp']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_muv_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
             'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
             'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_sider_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['Hepatobiliary disorders',
             'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
             'Investigations', 'Musculoskeletal and connective tissue disorders',
             'Gastrointestinal disorders', 'Social circumstances',
             'Immune system disorders', 'Reproductive system and breast disorders',
             'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
             'General disorders and administration site conditions',
             'Endocrine disorders', 'Surgical and medical procedures',
             'Vascular disorders', 'Blood and lymphatic system disorders',
             'Skin and subcutaneous tissue disorders',
             'Congenital, familial and genetic disorders',
             'Infections and infestations',
             'Respiratory, thoracic and mediastinal disorders',
             'Psychiatric disorders', 'Renal and urinary disorders',
             'Pregnancy, puerperium and perinatal conditions',
             'Ear and labyrinth disorders', 'Cardiac disorders',
             'Nervous system disorders',
             'Injury, poisoning and procedural complications']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_toxcast_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    # NB: some examples have multiple species, some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    tasks = list(input_df.columns)[1:]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
        labels.values


def _load_qm9_dataset(input_path):
    """

        :param input_path:
        :return: list of smiles, list of rdkit mol obj, np.array containing the
        labels
        """
    # NB: some examples have multiple species, some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    tasks = list(input_df.columns)[3:]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
        labels.values

def _load_chembl_with_labels_dataset(root_path):
    """
    Data from 'Large-scale comparison of machine learning methods for drug target prediction on ChEMBL'
    :param root_path: path to the folder containing the reduced chembl dataset
    :return: list of smiles, preprocessed rdkit mol obj list, list of np.array
    containing indices for each of the 3 folds, np.array containing the labels
    """
    # adapted from https://github.com/ml-jku/lsc/blob/master/pythonCode/lstm/loadData.py
    # first need to download the files and unzip:
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced.zip
    # unzip and rename to chembl_with_labels
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Smiles.pckl
    # into the dataPythonReduced directory
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20LSTM.pckl

    # 1. load folds and labels
    f = open(os.path.join(root_path, 'folds0.pckl'), 'rb')
    folds = pickle.load(f)
    f.close()

    f = open(os.path.join(root_path, 'labelsHard.pckl'), 'rb')
    targetMat = pickle.load(f)
    sampleAnnInd = pickle.load(f)
    targetAnnInd = pickle.load(f)
    f.close()

    targetMat = targetMat
    targetMat = targetMat.copy().tocsr()
    targetMat.sort_indices()
    targetAnnInd = targetAnnInd
    targetAnnInd = targetAnnInd - targetAnnInd.min()

    folds = [np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
    targetMatTransposed = targetMat[sampleAnnInd[list(chain(*folds))]].T.tocsr()
    targetMatTransposed.sort_indices()
    # # num positive examples in each of the 1310 targets
    trainPosOverall = np.array([np.sum(targetMatTransposed[x].data > 0.5) for x in range(targetMatTransposed.shape[0])])
    # # num negative examples in each of the 1310 targets
    trainNegOverall = np.array(
        [np.sum(targetMatTransposed[x].data < -0.5) for x in range(targetMatTransposed.shape[0])])
    # dense array containing the labels for the 456331 molecules and 1310 targets
    denseOutputData = targetMat.A  # possible values are {-1, 0, 1}

    # 2. load structures
    f = open(os.path.join(root_path, 'chembl20LSTM.pckl'), 'rb')
    rdkitArr = pickle.load(f)
    f.close()

    assert len(rdkitArr) == denseOutputData.shape[0]
    assert len(rdkitArr) == len(folds[0]) + len(folds[1]) + len(folds[2])

    preprocessed_rdkitArr = []
    logger.info('preprocessing')
    for i in tqdm(range(len(rdkitArr))):
        m = rdkitArr[i]
        if m == None:
            preprocessed_rdkitArr.append(None)
        else:
            mol_species_list = split_rdkit_mol_obj(m)
            if len(mol_species_list) == 0:
                preprocessed_rdkitArr.append(None)
            else:
                largest_mol = get_largest_mol(mol_species_list)
                if len(largest_mol.GetAtoms()) <= 2:
                    preprocessed_rdkitArr.append(None)
                else:
                    preprocessed_rdkitArr.append(largest_mol)

    assert len(preprocessed_rdkitArr) == denseOutputData.shape[0]

    smiles_list = [AllChem.MolToSmiles(m) if m != None else None for m in
                   preprocessed_rdkitArr]  # bc some empty mol in the
    # rdkitArr zzz...

    assert len(preprocessed_rdkitArr) == len(smiles_list)

    return smiles_list, preprocessed_rdkitArr, folds, denseOutputData


# root_path = 'dataset/chembl_with_labels'

def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]


def create_all_datasets():
    #### create dataset
    downstream_dir = [
        'bace',
        'bbbp',
        'clintox',
        'esol',
        'freesolv',
        'hiv',
        'lipophilicity',
        'muv',
        'sider',
        'tox21',
        'toxcast'
    ]

    for dataset_name in downstream_dir:
        logger.info(dataset_name)
        root = "E:\Code\Molecule\MultiMol\data\chem_dataset\\" + dataset_name
        os.makedirs(root + "/processed", exist_ok=True)
        dataset = MoleculeDataset(dataset=dataset_name, root=root)
        logger.info(dataset)

    # dataset = MoleculeDataset(root="dataset/chembl_filtered", dataset="chembl_filtered")
    # logger.info(dataset)
    # dataset = MoleculeDataset(root="dataset/zinc_standard_agent", dataset="zinc_standard_agent")
    # logger.info(dataset)


# test MoleculeDataset object
if __name__ == "__main__":
    create_all_datasets()

