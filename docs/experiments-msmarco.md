# MS MARCO

## Inference
Get data and checkpoint from [Google Drive](https://drive.google.com/drive/folders/1w8_8kFlQaIsi-zfbh6yBaPGpK3_vLAZ6?usp=sharing)

Get MS MARCO collection.
```
wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz -P ./data
tar -zxvf ./data/collection.tar.gz -C ./data/
```

Reproduce bert-base, MRR@10(dev): 0.3494.
```
CUDA_VISIBLE_DEVICES=0 \
python inference.py \
        -task ranking \
        -model bert \
        -max_input 12800000 \
        -test queries=./data/queries.dev.small.tsv,docs=./data/collection.tsv,trec=./data/run.msmarco-passage.dev.small.trec \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -checkpoint ./checkpoints/bert-base.bin \
        -res ./results/bert-base_msmarco-dev.trec \
        -max_query_len 32 \
        -max_doc_len 221 \
        -batch_size 256
```

Reproduce electra-base, MRR@10(dev): 0.3518.
```
CUDA_VISIBLE_DEVICES=0 \
python inference.py \
        -task ranking \
        -model bert \
        -max_input 12800000 \
        -test queries=./data/queries.dev.small.tsv,docs=./data/collection.tsv,trec=./data/run.msmarco-passage.dev.small.trec \
        -vocab google/electra-base-discriminator \
        -pretrain google/electra-base-discriminator \
        -checkpoint ./checkpoints/electra-base.bin \
        -res ./results/electra-base_msmarco-dev.trec \
        -max_query_len 32 \
        -max_doc_len 221 \
        -batch_size 256
```

For eval dataset inference, just change the trec file to *./data/run.msmarco-passage.eval.small.trec*. The top1000 trec files for dev and eval queries are generated following [anserini](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md).

## train
Train.
```
CUDA_VISIBLE_DEVICES=0 \
python train.py \
        -task ranking \
        -model bert \
        -train  queries=./data/queries.train.small.tsv,docs=./data/collection.tsv,qrels=./data/qrels.train.tsv,trec=./data/trids_bm25_marco-10.tsv \
        -max_input 12800000 \
        -save ./checkpoints/bert.bin \
        -dev queries=./data/queries.dev.small.tsv,docs=./data/collection.tsv,qrels=./data/qrels.dev.small.tsv,trec=./data/run.msmarco-passage.dev.small.100.trec \
        -qrels ./data/qrels.dev.small.tsv \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -res ./results/bert.trec \
        -metric mrr_cut_10 \
        -max_query_len 32 \
        -max_doc_len 221 \
        -epoch 3 \
        -batch_size 16 \
        -lr 3e-6 \
        -n_warmup_steps 160000 \
        -eval_every 10000
```

Since the whole dev dataset is too large, we only evaluate on top100 when training, and inference on whole dataset.
