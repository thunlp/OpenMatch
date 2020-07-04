# OpenMatch
An Open-Source Package for OpenQA and IR.

## 😃 What's New
* **[Top Spot on TREC-COVID Challenge](https://ir.nist.gov/covidSubmit/about.html)** (May 2020, Round2)

  The twin goals of the challenge are to evaluate search algorithms and systems for helping scientists, clinicians, policy makers, and others manage the existing and rapidly growing corpus of scientific literature related to COVID-19, and to discover methods that will assist with managing scientific information in future global biomedical crises. \
  [>> Reproduce Our Submit](./docs/experiments-treccovid.md) [>> About COVID-19 Dataset](https://www.semanticscholar.org/cord19)

## Overview
> **OpenMatch** integrates excellent neural methods and technologies to provide a complete solution for deep text matching and understanding.

* **Document Retrieval**

  TBD

* **Passage Reranking**

  Passage Reranking aims to produce a relevance ranked list of passages by matching texts against user queries.

* **Knowledge Enhancing**

  TBD

* **Data Augmentation**

  Data Augmentation leverages weak supervision data to improve the ranking accuracy in certain areas that lacks large scale relevance labels.

* **Learning-To-Rank**

  TBD

  |Stage|Model|Paper|
  |:----|:----:|:----|
  |Document Retrieval|**BM25**|TBD|
  |Document Retrieval|ANN|TBD|
  ||
  |Passage Reranking|**K-NRM**|[End-to-End Neural Ad-hoc Ranking with Kernel Pooling](https://dl.acm.org/doi/pdf/10.1145/3077136.3080809)|
  |Passage Reranking|Conv-KNRM|[Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search](https://dl.acm.org/doi/pdf/10.1145/3159652.3159659)|
  |Passage Reranking|TK|[Interpretable & Time-Budget-Constrained Contextualization for Re-Ranking](https://arxiv.org/pdf/1912.01385.pdf)|
  |Passage Reranking|BERT|[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)|
  ||
  |Knowledge Enhancing|**EDRM**|[Entity-Duet Neural Ranking: Understanding the Role of Knowledge Graph Semantics in Neural Information Retrieval](https://arxiv.org/pdf/1805.07591.pdf)|
  ||
  |Data Augmentation|**ReInfoSelect**|[Selective Weak Supervision for Neural Information Retrieval](https://arxiv.org/pdf/2001.10382v1.pdf)|
  ||
  |Learning-To-Rank|**Coordinate Ascent**|TBD|

Note that the BERT model is following huggingface's implementation - [transformers](https://github.com/huggingface/transformers), so other bert-like models are also available in our toolkit, e.g. electra, scibert.

## Installation
```
pip install git+https://github.com/thunlp/OpenMatch.git
```
### From Source
```
git clone https://github.com/thunlp/OpenMatch.git
cd OpenMatch
python setup.py install
```

## Quick Start
Detailed examples are available [here](./docs/openmatch.md).

```python
import torch
import OpenMatch as om

query = "Classification treatment COVID-19"
doc = "By retrospectively tracking the dynamic changes of LYM% in death cases and cured cases, this study suggests that lymphocyte count is an effective and reliable indicator for disease classification and prognosis in COVID-19 patients."
```

For bert-like models:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
input_ids = tokenizer.encode(query, doc)
model = om.models.Bert("allenai/scibert_scivocab_uncased")
ranking_score, ranking_features = model(torch.tensor(input_ids).unsqueeze(0))
```

For other models:
```python
tokenizer = om.data.tokenizers.WordTokenizer(pretrained="./data/glove.6B.300d.txt")
query_ids, query_masks = tokenizer.process(query, max_len=16)
doc_ids, doc_masks = tokenizer.process(doc, max_len=128)
model = om.models.KNRM(vocab_size=tokenizer.get_vocab_size(),
                       embed_dim=tokenizer.get_embed_dim(),
                       embed_matrix=tokenizer.get_embed_matrix())
ranking_score, ranking_features = model(torch.tensor(query_ids).unsqueeze(0),
                                        torch.tensor(query_masks).unsqueeze(0),
                                        torch.tensor(doc_ids).unsqueeze(0),
                                        torch.tensor(doc_masks).unsqueeze(0))
```

The GloVe can be downloaded using:
```
wget http://nlp.stanford.edu/data/glove.6B.zip -P ./data
unzip ./data/glove.6B.zip -d ./data
```

## Experiments
* All results is measured on ndcg@20 with 5 fold cross-validation.

|Model|ClueWeb09|Robust04|ClueWeb12|
|:---:|:-------:|:------:|:-------:|
|KNRM|0.1880|0.3016|0.0968|
|Conv-KNRM|0.1894|0.2907|0.0896|
|EDRM|0.2015|0.2993|0.0937|
|TK|0.2306|0.2822|0.0966|
|BERT|0.2701|0.4168|0.1183|
|ELECTRA|0.2861|0.4668|0.1078|

## TBD
ANN

## Contribution
Thanks to all the people who contributed to OpenMatch!

[Kaitao Zhang](https://github.com/zkt12), [Aowei Lu](https://github.com/LAW991224), [Si Sun](https://github.com/SunSiShining), [Zhenghao Liu](http://nlp.csai.tsinghua.edu.cn/~lzh/)

## Project Organizers
- Zhiyuan Liu
  * Tsinghua University
  * [Homepage](http://nlp.csai.tsinghua.edu.cn/~lzy/)
- Chenyan Xiong
  * Microsoft Research AI
  * [Homepage](https://www.microsoft.com/en-us/research/people/cxiong/)
