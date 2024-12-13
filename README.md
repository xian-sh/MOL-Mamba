# MOL-Mamba: Enhancing Molecular Representation with Structural & Electronic Insights

The paper is accepted by AAAI 2025.

_Hefei University of Technology, Hefei Comprehensive National Science Center, Anhui Zhonghuitong Technology Co., Ltd._

#### [\[arXiv\]](xxx) | [\[PDF\]](xx)



## Abstract

Molecular representation learning plays a crucial role in various downstream tasks, such as molecular property prediction and drug design. To accurately represent molecules, Graph Neural Networks (GNNs) and Graph Transformers (GTs) have shown potential in the realm of self-supervised pre-training. However, existing approaches often overlook the relationship between molecular structure and electronic information, as well as the internal semantic reasoning within molecules. This omission of fundamental chemical knowledge in graph semantics leads to incomplete molecular representations, missing the integration of structural and electronic data. To address these issues, we introduce MOL-Mamba, a framework that enhances molecular representation by combining structural and electronic insights. MOL-Mamba consists of an Atom & Fragment Mamba-Graph (MG) for hierarchical structural reasoning and a Mamba-Transformer (MT) fuser for integrating molecular structure and electronic correlation learning. Additionally, we propose a Structural Distribution Collaborative Training and E-semantic Fusion Training framework to further enhance molecular representation learning. Extensive experiments demonstrate that MOL-Mamba outperforms state-of-the-art baselines across eleven chemical-biological molecular datasets. 


## Approach

![test](assert/fig_mamba-fusion1_01.png)

---

## Data preparation

We need to prepare data before either pretraining or finetuning. This process will create and store a molecular graph and a fragment graph for each molecule based on a vocabulary of fragments.

```
 cd datasets 
 python loader_pretrain.py --root <output data path> --data_file_path <raw data path>
```

## Env

```
pip install rdkit-pypi
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda config --add channels pytorch
conda install pytorch-geometric -c rusty1s -c conda-forge
```

## 🌐 Usage

### ⚙ Network Architecture

Our EulerMormer is implemented in `model/magnet.py`.

- For **Config:** `config.py`

- For **train:** `python main.py`

- For **test video:** `python test_video.py`

[Demo Baby](https://github.com/VUT-HFUT/EulerMormer/blob/main/fig/baby.avi)
[Demo Drum](https://github.com/VUT-HFUT/EulerMormer/blob/main/fig/drum.avi)
[Demo Cattoy](https://github.com/VUT-HFUT/EulerMormer/blob/main/fig/cattoy.avi)

## 🔖:Citation

If you found this code useful please consider citing our [paper](xxxx):

    xxx
