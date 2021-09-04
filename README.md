# OpenMatch
An Open-Source Package for Information Retrieval.

## 😃 What's New
* **[Top Spot on TREC-COVID Challenge](https://ir.nist.gov/covidSubmit/about.html)** (May 2020, Round2)

  The twin goals of the challenge are to evaluate search algorithms and systems for helping scientists, clinicians, policy makers, and others manage the existing and rapidly growing corpus of scientific literature related to COVID-19, and to discover methods that will assist with managing scientific information in future global biomedical crises. \
  [>> Reproduce Our Submit](./docs/experiments-treccovid.md) [>> About COVID-19 Dataset](https://www.semanticscholar.org/cord19) [>> Our Paper](https://arxiv.org/abs/2011.01580)

## Overview
**OpenMatch** integrates excellent neural methods and technologies to provide a complete solution for deep text matching and understanding. The documentation and tutorial of OpenMatch are available at [here](https://openmatch-docs.readthedocs.io/en/latest/index.html).

>### 1/ Document Retrieval

  Document Retrieval refers to extracting a set of related documents from large-scale document-level data based on user queries.

### **\* Sparse Retrieval**

Sparse Retriever is defined as a sparse bag-of-words retrieval model.

### **\* Dense Retrieval**

Dense Retriever performs retrieval by encoding documents and queries into dense low-dimensional vectors, and selecting the document that has the highest inner product with the query

>### 2/ Document Reranking

Document reranking aims to further match user query and documents retrieved by the previous step with the purpose of obtaining a ranked list of relevant documents.

### **\* Neural Ranker**

Neural Ranker uses neural network as ranker to reorder documents.


### **\* Feature Ensemble**

Feature Ensemble can fuse neural features learned by neural ranker with the features of non-neural methods to obtain more robust performance

>### 3/ Domain Transfer Learning

  Domain Transfer Learning can leverages external knowledge graphs or weak supervision data to guide and help ranker to overcome data scarcity.

### **\* Knowledge Enhancemnet**

  Knowledge Enhancement incorporates entity semantics of external knowledge graphs to enhance neural ranker.

### **\* Data Augmentation**

  Data Augmentation leverages weak supervision data to improve the ranking accuracy in certain areas that lacks large scale relevance labels.


  |Stage|Model|Paper|
  |:----|:----:|:----|
  |1/ Sparse Retrieval|**BM25**|Best Match25 [~Tool](https://github.com/castorini/anserini)|
  |1/ Dense Retrieval|**ANN**|Approximate nearest neighbor [~Tool](https://github.com/facebookresearch/faiss)|
  ||
  |2/ Neural Ranker|**K-NRM**|End-to-End Neural Ad-hoc Ranking with Kernel Pooling [~Paper](https://dl.acm.org/doi/pdf/10.1145/3077136.3080809)|
  |2/ Neural Ranker|**Conv-KNRM**|Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search [~Paper](https://dl.acm.org/doi/pdf/10.1145/3159652.3159659)|
  |2/ Neural Ranker|**TK**|Interpretable & Time-Budget-Constrained Contextualization for Re-Ranking [~Paper](https://arxiv.org/pdf/2002.01854.pdf)|
  |2/ Neural Ranker|**BERT**|BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [~Paper](https://arxiv.org/pdf/1810.04805.pdf)|
  |2/ Feature Ensemble|**Coordinate Ascent**|Linear feature-based models for information retrieval. Information Retrieval [~Paper](https://lintool.github.io/Ivory/docs/publications/Metzler_Croft_2007.pdf)
  ||
  |3/ Knowledge Enhancement|**EDRM**|Entity-Duet Neural Ranking: Understanding the Role of Knowledge Graph Semantics in Neural Information Retrieval [~Paper](https://arxiv.org/pdf/1805.07591.pdf)|
  |3/ Data Augmentation|**ReInfoSelect**|Selective Weak Supervision for Neural Information Retrieval [~Paper](https://arxiv.org/pdf/2001.10382v1.pdf)|

  Note that the BERT model is following huggingface's implementation - [transformers](https://github.com/huggingface/transformers), so other bert-like models are also available in our toolkit, e.g. electra, scibert.

## Installation

#### \* From PyPI

```
pip install git+https://github.com/thunlp/OpenMatch.git
```

#### \* From Source
```
git clone https://github.com/thunlp/OpenMatch.git
cd OpenMatch
python setup.py install
```

#### \* From Docker
To build an OpenMatch docker image from Dockerfile 
```
docker build -t <image_name> .
```

To run your docker image just built above as a container
```
docker run --gpus all --name=<container_name> -it -v /:/all/ --rm <image_name>:<TAG>
```

## Quick Start

\*  Detailed examples are available [here](./docs/openmatch.md).

```python
import torch
import OpenMatch as om

query = "Classification treatment COVID-19"
doc = "By retrospectively tracking the dynamic changes of LYM% in death cases and cured cases, this study suggests that lymphocyte count is an effective and reliable indicator for disease classification and prognosis in COVID-19 patients."
```

\*  For bert-like models:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
input_ids = tokenizer.encode(query, doc)
model = om.models.Bert("allenai/scibert_scivocab_uncased")
ranking_score, ranking_features = model(torch.tensor(input_ids).unsqueeze(0))
```

\*  For other models:

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

\*  The GloVe can be downloaded using:

```
wget http://nlp.stanford.edu/data/glove.6B.zip -P ./data
unzip ./data/glove.6B.zip -d ./data
```

\* Evaluation

```
metric = om.Metric()
res = metric.get_metric(qrels, ranking_list, 'ndcg_cut_20')
res = metric.get_mrr(qrels, ranking_list, 'mrr_cut_10')
```

## Experiments
\* [Ad-hoc Search](./docs/experiments-adhoc.md)

  |Retriever|Reranker|Coor-Ascent|ClueWeb09|Robust04|ClueWeb12|
  |:-------:|:------:|:---------:|:-------:|:------:|:-------:|
  |SDM|KNRM|-|0.1880|0.3016|0.0968|
  |SDM|Conv-KNRM|-|0.1894|0.2907|0.0896|
  |SDM|EDRM|-|0.2015|0.2993|0.0937|
  |SDM|TK|-|0.2306|0.2822|0.0966|
  |SDM|BERT Base|-|0.2701|0.4168|0.1183|
  |SDM|ELECTRA Base|-|0.2861|0.4668|0.1078|

\* [MS MARCO Passage Ranking](./docs/experiments-msmarco.md)

  |Retriever|Reranker|Coor-Ascent|dev|eval|
  |:-------:|:------:|:---------:|:-:|:--:|
  |BM25|BERT Base|-|0.349|0.345|
  |BM25|ELECTRA Base|-|0.352|0.344|
  |BM25|RoBERTa Large|-|0.386|0.375|
  |BM25|ELECTRA Large|-|0.388|0.376|

\* [MS MARCO Document Ranking](./docs/experiments-msmarco-doc.md)

  |Retriever|Reranker|Coor-Ascent|dev|eval|
  |:-------:|:------:|:---------:|:-:|:--:|
  |ANCE FirstP|-|-|0.373|0.334|
  |ANCE MaxP|-|-|0.383|0.342|
  |ANCE FirstP+BM25|BERT Base FirstP|+|0.431|0.380|
  |ANCE MaxP|BERT Base MaxP|+|0.432|0.391|

\* [Classic Features](./docs/experiments-classic.md)

  <table>
      <tr>
          <td rowspan="2">Methods</td>
          <td colspan="2">ClueWeb09-B</td>
          <td colspan="2">Robust04</td>
          <td colspan="2">TREC-COVID</td>
    </tr>
      <tr>
          <td>NDCG@20</td>
          <td>ERR@20</td>
          <td>NDCG@20</td>
          <td>ERR@20</td>
          <td>NDCG@20</td>
          <td>P@20</td>
    </tr>
      <tr>
          <td>BM25 (Anserini)</td>
          <td>0.2773</td>
          <td>0.1426</td>
          <td>0.4129</td>
          <td>0.1117</td>
          <td>0.6979</td>
          <td>0.7670</td>
      </tr>
      <tr>
          <td>RankSVM (Dai et al.)</td>
          <td>0.289</td>
          <td>n.a.</td>
          <td>0.420</td>
          <td>n.a.</td>
          <td>n.a.</td>
          <td>n.a.</td>
      </tr>
      <tr>
          <td>RankSVM (OpenMatch)</td>
          <td>0.2825</td>
          <td>0.1476</td>
          <td>0.4309</td>
          <td>0.1173</td>
          <td>0.6995</td>
          <td>0.7570</td>
      </tr>
      <tr>
          <td>Coor-Ascent (Dai et al.)</td>
          <td>0.295</td>
          <td>n.a.</td>
          <td>0.427</td>
          <td>n.a.</td>
          <td>n.a.</td>
          <td>n.a.</td>
      </tr>
      <tr>
          <td>Coor-Ascent (OpenMatch)</td>
          <td>0.2969</td>
          <td>0.1581</td>
          <td>0.4340</td>
          <td>0.1171</td>
          <td>0.7041</td>
          <td>0.7770</td>
      </tr>
  </table>

## Contribution
Thanks to all the people who contributed to OpenMatch!

[Kaitao Zhang](https://github.com/zkt12), [Si Sun](https://github.com/SunSiShining), [Zhenghao Liu](http://nlp.csai.tsinghua.edu.cn/~lzh/), [Aowei Lu](https://github.com/LAW991224)

## Project Organizers
- Zhiyuan Liu
  * Tsinghua University
  * [Homepage](http://nlp.csai.tsinghua.edu.cn/~lzy/)
- Chenyan Xiong
  * Microsoft Research AI
  * [Homepage](https://www.microsoft.com/en-us/research/people/cxiong/)
- Maosong Sun
  * Tsinghua University
  * [Homepage](http://nlp.csai.tsinghua.edu.cn/staff/sms/)
