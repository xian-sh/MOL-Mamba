import torch
from dataclasses import dataclass, asdict
# from datasets.utils.logger import setup_logger, dict_to_markdown
import yaml


# classification  BBBP, Tox21, SIDER,      ToxCast,  BACE, ClinTox, HIV, MUV
# regression ESOL, FreeSolv, Lipophilicity

dataset = 'BBBP'
task_type = 'classification'
split = 'scaffold'  # 'scaffold' 'random'
dim = 256


@dataclass
class ModelArgs:
    # task
    dataset: str = dataset.lower()
    split: str = split
    task_type: str = task_type
    save_dir: str = f'./result_{task_type}/{dataset.lower()}/{split}/all/'
    data_dir: str = './data/chem_dataset/'
    model: str = 'MSE'
    pretrain_path: str = None

    # training settings
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    runseed: int = 0
    seed: int = 42
    num_workers: int = 0
    epoch: int = 10
    batchsize: int = 64
    lr: float = 0.0001
    decay: float = 0.0
    drop: int = 0.1

    # module settings
    use_gnn: bool = True
    use_frag_gnn: bool = True
    use_graph_mamba: bool = True
    use_fusion: bool = True
    mask_e: bool = False

    # gnn encoder settings
    gnn_layer: int = 6  # 6
    emb_dim: int = dim
    frag_layer: int = 6

    # graph_mamba settings
    d_model: int = dim
    n_layer: int = 4
    sch_layer: int = 6
    dim_in: int = 2
    cutoff: float = 10.0

    # elec settings
    d_in: int = 1
    d_h: int = dim
    d_o: int = dim
    mask_ratio: float = 0.1
    # e_layer: int = 4

    # loss weight
    w_f_d: float = 0.01
    w_frag: float = 5.0
    w_tree: float = 0.05
    w_task: float = 1.0
    w_mask: float = 0.001

    def to_dict(self):
        return asdict(self)

    def save_yaml(self, filename):
        with open(filename, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filename):
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)


cfg = ModelArgs()

