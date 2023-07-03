# Curriculum Learning for Graph Neural Networks: A Multiview Competence-based Approach 

Multiview Competence-based Curriculum Learning (MCCL), is a new perspective on curriculum learning introducing a novel approach that builds on graph complexity formalisms (as difficulty criteria) and model competence during training. The model consists of a scheduling scheme which derives effective curricula by accounting for different views of sample difficulty. It effectively leverages complexity formalisms of graph data, taking into account multiview difficulty of training data samples and model competency.

<p align="center">
<img src="https://github.com/CLU-UML/MCCL/blob/main/mccl.png" width="800" height="450">
</p>
The difficulty of training examples--target node pairs in red--can be assessed based on their subgraphs, k-hop neighbors of target nodes. For brevity, we show two views: node degree and closeness centrality. Boldfaced tokens indicate nodes in the graph, Label indicates if the sentence reports a causal relation between the nodes, and Degree and Centrality report the sum of degree and closeness centrality scores of the target nodes in their subgraphs. Each subgraph provides a structural view of the target nodes in sentences. The relative difficulty of examples is different across views, e.g., G2 is less difficult than G3 according to Degree but more difficult according to Centrality.


<br>For node classification, input to the model is the graph with text features and pre-calculated graph complexity indices (example, average degree, number of nodes, etc.), and the output is the multiclass label. For link prediction, input to the model is a graph with text features generated from textual summaries and pre-calculated graph complexity indices (example, average degree, number of nodes, etc.), and the output is a binary label (+1/-1). <br />

# Data 

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

### Link Prediction
There are two datasets for link prediction: PGR and GDPR. 

* **Phenotype Gene Relation (PGR):**  PGR is created by Sousa et al., NAACL 2019 (https://aclanthology.org/N19-1152/) from PubMed articles and contains sentences describing relations between given genes and phenotypes. In our experiments, we only include data samples in PGR with available text descriptions for their genes and phenotypes. This amounts to ~71% of the original dataset. 

* **Gene, Disease, Phenotype Relation (GDPR):** This dataset is obtained by combining and linking entities across two freely-available datasets: Online Mendelian Inheritance in Man (OMIM, https://omim.org/) and Human Phenotype Ontology (HPO, https://hpo.jax.org/). The dataset contains relations between genes, diseases and phenotypes.

To download datasets with embeddings and Train/Test/Val splits, go to data directory and run download.sh as follows

```
sh ./download.sh
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
