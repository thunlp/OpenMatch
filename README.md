# OpenMatch
An Open-Source Package for OpenQA and IR.

## Installation
```
pip install git+https://github.com/thunlp/OpenMatch.git
```

## Quick Start
For bert training
```
sh train_bert.sh
```

For edrm, cknrm, knrm or tk training
```
sh train.sh
```

For bert inference
```
sh inference_bert.sh
```

For edrm, cknrm, knrm or tk inference
```
sh inference.sh
```

More information is available [here](./docs/openmatch.md).

## Experiments
* [TREC-COVID Challenge](./docs/experiments-treccovid.md)
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
