# OpenMatch

## Train
```
-task             'ranking': pair-wise, 'classification': query-doc.
-model            'bert', 'tk', 'cknrm' or 'knrm'.
-reinfoselect     use reinfoselect or not.
-train            path to training dataset.
-max_input        max input of instances.
-save             path for saving model checkpoint.
-dev              path to dev dataset.
-qrels            path to qrels.
-vocab            path to vocab.
-pretrain         path to pretrained bert model.
-res              path for saving result.
-metric           which metrics to use, e.g. ndcg_cut_10.
-n_kernels        kernel number, for tk, cknrm or knrm.
-max_query_len    max length of query.
-max_doc_len      max length of document.
-epoch            how many epoch.
-batch_size       batch size.
-lr               learning rate.
-eval_every       e.g. 1000, every 1000 steps evaluate on dev data.
```

## Inference
```
-task             'ranking': pair-wise, 'classification': query-doc.
-model            'bert', 'tk', 'cknrm' or 'knrm'.
-max_input        max input of instances.
-test             path to test dataset.
-vocab            path to vocab.
-pretrain         path to pretrained bert model.
-checkpoint       path to checkpoint.
-res              path for saving result.
-n_kernels        kernel number, for tk, cknrm or knrm.
-max_query_len    max length of query.
-max_doc_len      max length of document.
-batch_size       batch size.
```
