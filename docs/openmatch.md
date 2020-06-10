# OpenMatch

## Options
### Train
```
-task             'ranking': pair-wise, 'classification': query-doc.
-model            'bert', 'tk', 'edrm', 'cknrm' or 'knrm'.
-reinfoselect     use reinfoselect or not.
-train            path to training dataset.
-max_input        max input of instances.
-save             path for saving model checkpoint.
-dev              path to dev dataset.
-qrels            path to qrels.
-vocab            path to glove or customized vocab.
-ent_vocab        path to entity vocab, for edrm.
-pretrain         path to pretrained bert model.
-res              path for saving result.
-metric           which metrics to use, e.g. ndcg_cut_10.
-n_kernels        kernel number, for tk, edrm, cknrm or knrm.
-max_query_len    max length of query tokens.
-max_doc_len      max length of document tokens.
-epoch            how many epoch.
-batch_size       batch size.
-lr               learning rate.
-eval_every       e.g. 1000, every 1000 steps evaluate on dev data.
```

### Inference
```
-task             'ranking': pair-wise, 'classification': query-doc.
-model            'bert', 'tk', 'edrm', 'cknrm' or 'knrm'.
-max_input        max input of instances.
-test             path to test dataset.
-vocab            path to glove or customized vocab.
-ent_vocab        path to entity vocab.
-pretrain         path to pretrained bert model.
-checkpoint       path to checkpoint.
-res              path for saving result.
-n_kernels        kernel number, for tk, edrm, cknrm or knrm.
-max_query_len    max length of query tokens.
-max_doc_len      max length of document tokens.
-batch_size       batch size.
```

## Data Format
### Ranking Task
For bert, tk, cknrm or knrm:

|file|format|
|:---|:-----|
|train|{"query": str, "doc\_pos": str, "doc\_neg": str}|
|dev  |{"query": str, "doc": str, "label": int, "query\_id": str, "paper\_id": str, "retrieval\_score": float}|
|test |{"query": str, "doc": str, "query\_id": str, "paper\_id": str, "retrieval\_score": float}|

For edrm:

|file|format|
|:---|:-----|
|train|+{"query\_ent": list, "doc\_pos\_ent": list, "doc\_neg\_ent": list, "query\_des": list, "doc\_pos\_des": list, "doc\_neg\_des": list}|
|dev  |+{"query\_ent": list, "doc\_ent": list, "query\_des": list, "doc\_des": list}|
|test |+{"query\_ent": list, "doc\_ent": list, "query\_des": list, "doc\_des": list}|

The *query_ent*, *doc_ent* is a list of entities relevant to the query or document, *query_des* is a list of entity descriptions.

### Classification Task
Only train file format different with ranking task.

For bert, tk, cknrm or knrm:

|file|format|
|:---|:-----|
|train|{"query": str, "doc": str, "label": int}|

For edrm:

|file|format|
|:---|:-----|
|train|+{"query\_ent": list, "doc\_ent": list, "query\_des": list, "doc\_des": list}|

### Others
The dev and test files can be set as:
```
-dev queries={path to queries},docs={path to docs},qrels={path to qrels},trec={path to trec}
-test queries={path to queries},docs={path to docs},trec={path to trec}
```

|file|format|
|:---|:-----|
|queries|{"query\_id":, "query":}|
|docs|{"doc\_id":, "doc":}|
|qrels|query\_id iteration doc\_id label|
|trec|query\_id Q0 doc\_id rank score run-tag|

For edrm, the queries and docs is a little different:

|file|format|
|:---|:-----|
|queries|+{"query\_ent": list, "query\_des": list}|
|docs|+{"doc\_ent": list, "doc\_des": list}|
