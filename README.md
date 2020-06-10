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

Options
```
-task            choices=['ranking', 'classification']
-model           choices=['bert', 'tk', 'edrm', 'cknrm', 'knrm']
```
More information is available [here](./docs/openmatch.md).

## OpenMatch Experiments
* [TREC-COVID Challenge](./docs/experiments-treccovid.md)

## Todo List
ANN
