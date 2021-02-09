# Hierarchical-Graph-Network-for-Multi-hop-Question-Answering (HGN)

Hierarchical Graph Network (HGN) is a multi-hop reasoning pathway manifested in a hierarchical graph structure. Built and proposed by Microsoft Dynamics 365 AI Research, HGN aggregates clues from sources of differing granularity levels (e.g., paragraphs, sentences, entities). It effectively incorporates concepts from Graph Attention Network (GAT), Gated Attention and BiDAF (Seo et al., 2017) to construct a multi-hop reasoning graph network model on HotpotQA.

I have implemented it with the [Deep Graph Library (DGL)](https://www.dgl.ai/), which provides well-implemented versions of varous graph algorithms (e.g., GCN, GraphSAGE, GAT, Jumping Knowledge Network, etc.). By using `dgl.heterograph` and `dgl.nn.pytorch.HeteroGraphConv`, I was easily able to construct nodes of differing granularity levels and their update algorithm.

- Dataset: HotpotQA 

## Dependencies

- torch==1.4.0
- transformers==2.10.0
- dgl==0.5.2
- spacy==2.2.3

## Usage

Training the paragraph selector (fine-tune the paragraph selector for paragraph retrieval):
```bash
$ ./para_finetune.sh
```

Training the HGN model:
```bash
$ ./train.sh --do_train
```

Add `--do_eval` for evaluation.

## References
- [Hierarchical Graph Network (paper)](https://arxiv.org/pdf/1911.03631.pdf)
