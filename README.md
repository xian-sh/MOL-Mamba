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

### Pretraining data: Geometric Ensemble Of Molecules (GEOM)

```bash
mkdir -p GEOM/raw
mkdir -p GEOM/processed
```

- GEOM: [Paper](https://arxiv.org/pdf/2006.05531v3.pdf), [GitHub](https://github.com/learningmatter-mit/geom)

- Data Download:
  
  ```bash
  wget https://dataverse.harvard.edu/api/access/datafile/4327252
  mv 4327252 rdkit_folder.tar.gz
  tar -xvf rdkit_folder.tar.gz
  ```
  
  or do the following if you are using slurm system

      cp rdkit_folder.tar.gz $SLURM_TMPDIR
      cd $SLURM_TMPDIR
      tar -xvf rdkit_folder.tar.gz

- over 430k molecules

  - 304,466 species contain experimental data for the inhibition of various pathogens

  - 133,258 are species from the QM9

### Chem Dataset

```
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
mv dataset molecule_datasets
```

### Making Data

We need to prepare data before either pretraining or finetuning. This process will create and store a molecular graph and a fragment graph for each molecule based on a vocabulary of fragments. 

```
cd datasets
python loader_geom.py
python loader_downstream.py
```

## Env

```
pip install rdkit-pypi
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda config --add channels pytorch
conda install pytorch-geometric -c rusty1s -c conda-forge
```

## Training

```
python train_class.py
```

## LICENSE
Many parts of the implementations are borrowed from [GraphFP](https://github.com/lvkd84/GraphFP).
Our codes are under [MIT](https://opensource.org/licenses/MIT) license.
