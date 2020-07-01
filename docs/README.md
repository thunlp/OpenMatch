# OpenMatch
An Open-Source Package for Open-Domain QA (OpenQA) and Information Retrieval (IR).

**SS: 这里需要稍微更详细更吸引人的描述**

## 😃 News
* **[Top Spot on TREC-COVID Challenge](https://ir.nist.gov/covidSubmit/about.html)** (May 2020, Round2)

  The twin goals of the challenge are to evaluate search algorithms and systems for helping scientists, clinicians, policy makers, and others manage the existing and rapidly growing corpus of scientific literature related to COVID-19, and to discover methods that will assist with managing scientific information in future global biomedical crises. >> [About COVID-19 Dataset](https://www.semanticscholar.org/cord19) >> [Reproduce Our Submit]()

**SS: "Reproduce Our Submit"这里可以link到docs里专门介绍trec-covid的文件**

## Overview
**OpenMatch** integrates excellent neural methods and technologies to provide a complete solution for deep text matching and understanding.

* **Document Indexing**

> **Document Indexing** associates the information with a document allowing it to be easily retrieved later according to the user's query.

* **Document Retrieval**

> **Document Retrieval** aims to produce a relevance ranked list of documents by matching texts against user queries.

* **Question Answering**

> **Question Answering** locates precise answers to user queries from the related documents retrieved.

* **Data Augmentation**

> **Data Augmentation** leverages weak supervision data to improve the ranking accuracy in certain areas that lacks large scale relevance labels.


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

## Easy Start

### Start >> Document Indexing

* **Prepare Data**

**SS:给出如何进行数据预处理的步骤，或者给出data format**

* **Run Model**

  - Training

**SS: 这里给出如何利用👆上面准备的数据，来训练ANN model**


  - Inference

**SS: 这里给出如何利用👆上面准备的数据，来infer ANN model**

* **Evaluation**

**SS: 这里可以提供相应的测试步骤，来获得测评结果**


###  Start >> Document Retrieval

* **Prepare Data**

**SS: 给出如何进行数据预处理的步骤，或者给出data format (如果太多了可以把多余的部分放到doc里，增加跳转)**

* **Run Model**

  - Training

**SS: 这里给出如何利用👆上面准备的数据，选择不同的neural model来训练**

  - Inference

**SS: 这里给出如何利用👆上面准备的数据，来infer model**


* **Evaluation**

**SS: 这里可以提供相应的测试步骤，来获得测评结果**


###  Start >> Question Answering

**SS: QA我看工具包的介绍提到了，但是我不太清楚，是否需要详细的details**

* **Prepare Data**

* **Run Model**

  - Training


  - Inference

* **Evaluation**



### Start >> Data Augmentation

* **Prepare Data**

**SS: 如果数据预处理部分和测试部分和 Document Retrieval 相同，可以写 the same as Document Retrieval**

* **Run Model**

  - Training

  - Inference

* **Evaluation**


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
