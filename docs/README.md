# OpenMatch
An Open-Source Package for Open-Domain QA (OpenQA) and Information Retrieval (IR).

**SS: 这里需要稍微更详细更吸引人的描述**

## 😃 News
* **[Top Spot on TREC-COVID Challenge](https://ir.nist.gov/covidSubmit/about.html)** (May 2020, Round2)

  The twin goals of the challenge are to evaluate search algorithms and systems for helping scientists, clinicians, policy makers, and others manage the existing and rapidly growing corpus of scientific literature related to COVID-19, and to discover methods that will assist with managing scientific information in future global biomedical crises. \
  [>> Reproduce Our Submit]() [>> About COVID-19 Dataset](https://www.semanticscholar.org/cord19)


**SS: "Reproduce Our Submit"这里可以link到docs里专门介绍trec-covid的文件**

## Overview
> **OpenMatch** integrates excellent neural methods and technologies to
\
provide a complete solution for deep text matching and understanding.

* **Document Indexing**

  Document Indexing associates the information with a document allowing \
  it to be easily retrieved later according to the user's query.

* **Document Retrieval**

  Document Retrieval aims to produce a relevance ranked list of documents \
  by matching texts against user queries.

* **Question Answering**

  Question Answering locates precise answers to user queries from \
  the related documents retrieved.

* **Data Augmentation**

  Data Augmentation leverages weak supervision data to improve the ranking \
  accuracy in certain areas that lacks large scale relevance labels.


  |Stage|Models|Desription|
  |:----|:----:|:----|
  |Document Indexing|**ANN**|[Approximate Nearest Neighbor]()|
  ||
  |Document Retrieval|**K-NRM**|[Kernel-based Neural Ranking Model]()|
  |Document Retrieval|Conv-KNRM|
  |Document Retrieval|EDRM|[Entity-Duet Neural Ranking Model]()
  |Document Retrieval|TK|
  |Document Retrieval|BERT|
  |Document Retrieval|ELECTRA|
  ||
  |Question Answering|||
  ||
  |Data Augmentation|ReInfoSelect||

**SS: 填入模型的完整名称，添加相关paper link**


## Requirements & Installation

### Requirements

* **OpenMatch is dependent on Python and PyTorch**
  ```
  * Python == 3.7
  * PyTorch >= 1.0.0
  ```

* **Setup requirements directly**
  ```
  pip install -r requirements.txt
  ```

### Installation

- **Install OpenMatch from Pypi Package**

  ```
  pip install openmatch
  ```
- **Install OpenMatch from Github Source**

  ```
  git clone ....
  cd ...
  python setup.py install
  ```

**SS: 以上填入正确的安装步骤**


## Quick Start

### >> Document Indexing

* **Prepare Data**



* **Run Model**

  - Training


  - Inference



* **Evaluation**




### >> Document Retrieval

* **Prepare Data**


* **Run Model**

  - Training



  - Inference


* **Evaluation**




### >> Question Answering

* **Prepare Data**

* **Run Model**

  - Training


  - Inference

* **Evaluation**



### >> Data Augmentation

* **Prepare Data**


* **Run Model**

  - Training

  - Inference

* **Evaluation**


**SS: 在相应的位置填入较为详细的步骤**


## Experiments

  |Model|Dataset|Metrics|
  |:---:|:-----:|:-----:|
  ||||

  **SS: 将已有的实验结果整理到这里**

## Contribution

Thank you to all the people who contributed to OpenMatch !

[Kaitao Zhang](), ...,

**SS: 填入完整的贡献者，可以附上github或者homepage**

## Project Organizers

- **Zhiyuan Liu**
  * NLP Lab, Department of Computer Science, Tsinghua University
  * [Homepage](http://nlp.csai.tsinghua.edu.cn/~lzy/)


**SS: 填入完整的组织者，附上机构和homepage**
