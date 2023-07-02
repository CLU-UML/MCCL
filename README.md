# MCCL
Curriculum Learning for Graph Neural Networks: A Multiview Competence-based Approach
MCCL is a new perspective on curriculum learning by introducing a novel approach that builds on graph complexity formalisms (as difficulty criteria) and model competence during training. The model consists of a scheduling scheme which derives effective curricula by accounting for different views of sample difficulty and model competence during training. It effectively leverages complexity formalisms of graph data, taking into account multiview difficulty of training data samples and model's learning progress.
<p align="center">
<img src="https://github.com/CLU-UML/MCCL/blob/main/mccl.png" width="800" height="450">
</p>

# Data 
### Link Prediction
There are two datasets for link prediction: PGR and GDPR. 

* **Phenotype Gene Relation (PGR):**  PGR is created by Sousa et al., NAACL 2019 (https://aclanthology.org/N19-1152/) from PubMed articles and contains sentences describing relations between given genes and phenotypes. In our experiments, we only include data samples in PGR with available text descriptions for their genes and phenotypes. This amounts to ~71% of the original dataset. 

* **Gene, Disease, Phenotype Relation (GDPR):** This dataset is obtained by combining and linking entities across two freely-available datasets: Online Mendelian Inheritance in Man (OMIM, https://omim.org/) and Human Phenotype Ontology (HPO, https://hpo.jax.org/). The dataset contains relations between genes, diseases and phenotypes.

To download datasets with embeddings and Train/Test/Val splits, go to data directory and run download.sh as follows

```
sh ./download.sh
```
### Node Classification
There are two datasets for node classification: Arxiv and Cora. 

* **Arxiv:** Arxiv is downloaded from ogbn (https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) using following code:

```
from ogb.nodeproppred import PygNodePropPredDataset

dataset = PygNodePropPredDataset(name = d_name) 

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph = dataset[0] # pyg graph object

```
* **Cora:** Cora is downloaded from Pytorch Geometric library (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid). We used following code:

```
dataset = Planetoid(root='/tmp/Cora', name='Cora')
```
# To run the code 
Use the following command with appropriate arguments:
### Node Classification
```
python node_classification/node_prediction_CMCL.py
```
### Link Prediction
```
python edge_prediction/main.py
```
# Citation

```
@inproceedings{nidhi-etal-2023-cmcl,
    title = "Curriculum Learning for Graph Neural Networks: A Multiview Competence-based Approach",
    author = "Vakil, Nidhi and  Amiri, Hadi",
    booktitle = "Proceedings of the 2023 Association for Computational Linguistics",
    publisher = "Association for Computational Linguistics",
    year = "2023"
    
}
```
