# OpenMatch
An Open-Source Package for OpenQA and IR.

## Requirements
### Setup requirements directly
* `python == 3.7`
* `torch >= 1.0.0`

To run OpenMatch, please install all requirements.
```
pip install -r requirements.txt
```

Get glove embeddings.
```
wget http://nlp.stanford.edu/data/glove.6B.zip -P ./data
unzip ./data/glove.6B.zip -d ./data
```

## Data Format

|file|format|
|:---|:-----|
|train|{"query":, "doc\_pos":, "doc\_neg":}|
|dev  |{"query":, "doc":, "label":, "query\_id":, "paper\_id":, "retrieval\_score":}|
|test |{"query":, "doc":, "query\_id":, "paper\_id":, "retrieval\_score":}|

Or

|file|format|
|:---|:-----|
|queries|{"query\_id":, "query"}|
|docs|"doc\_id":, "doc"|
|qrels|query\_id iteration doc\_id label|
|trec|query\_id Q0 doc\_id rank score run-tag|

## Quick Start
For bert training
```
sh train_bert.sh
```

For cknrm, knrm or tk training
```
sh train.sh
```

For bert inference
```
sh inference_bert.sh
```

For cknrm, knrm or tk inference
```
sh inference.sh
```

## OpenMatch Experiments
* [TREC-COVID Challenge](./docs/experiments-treccovid.md)

## Todo List
### Models
EDRM

### Datasets
ReQA, WikiQA, SQuAD-Open
