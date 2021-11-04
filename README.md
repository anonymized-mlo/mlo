README
======

**exTV** (Model to **ex**tract **T**ie **V**alences) is a computational and data-driven approach to the problem of tie-valence prediction in networks involving hierarchy and competition.


Component
---------
exTV consists of 3 stages: NLP, GNN, Inference.

### NLP
NLP stage in exTV creates contextual email embeddings of ties with ELECTRA, LIWC, and VADER.

The code of NLP stage is in code/nlp.

![overall_model](https://user-images.githubusercontent.com/77777793/140386036-a7f5b93c-940e-4f1b-9ff8-2fb6eb0115ff.jpg)

### GNN
GNN stage composes a graph with employees in an organization, and aggregate their features based on following GNN techniques.

The code of GNN stage is in code/nlp/graph/best_signed_emb.py.
* SGCN
  - https://github.com/benedekrozemberczki/SGCN
* SLF
  - https://github.com/WHU-SNA/SLF
* SiGAT
  - https://github.com/huangjunjie-cs/SiGAT

### Inference
We finally infer the tie valence with email embeddings from NLP stage and employee embeddings from GNN stage.

The code of GNN stage is in code/graph/best_signed_with_edge.py.


Data
----
We do not provide training data for privacy reason.
