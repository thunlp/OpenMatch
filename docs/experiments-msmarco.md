# MS MARCO Passage Ranking
Given a query q and a the 1000 most relevant passages P = p1, p2, p3,... p1000, as retrieved by BM25 a successful system is expected to rerank the most relevant passage as high as possible. For this task not all 1000 relevant items have a human labeled relevant passage. Evaluation will be done using MRR. More details are available at [MSMARCO-Passage-Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking).

## Inference

Get data and checkpoint from [Google Drive](https://drive.google.com/drive/folders/1w8_8kFlQaIsi-zfbh6yBaPGpK3_vLAZ6?usp=sharing)

Get checkpoints of electra-large and roberta-large from [electra-large](https://drive.google.com/file/d/1e0FUHuzE4sEzWvoXLmcowY9P3_c6N1sk/view?usp=sharing) [roberta-large](https://drive.google.com/file/d/1fUBSSaYgYwKU6muKWqfsnAUCI98SUbpQ/view?usp=sharing)

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

Reproduce electra-large, MRR@10(dev): 0.388

```
CUDA_VISIBLE_DEVICES=0 \
python inference.py \
        -task ranking \
        -model bert \
        -max_input 12800000 \
        -test queries=./data/queries.dev.small.tsv,docs=./data/collection.tsv,trec=./data/run.msmarco-passage.dev.small.trec \
        -vocab google/electra-large-discriminator \
        -pretrain google/electra-large-discriminator \
        -checkpoint ./checkpoints/electra_large.bin \
        -res ./results/electra-large_msmarco-dev.trec \
        -max_query_len 32 \
        -max_doc_len 221 \
        -batch_size 256
```

Reproduce roberta-large, MRR@10(dev): 0.386

```
CUDA_VISIBLE_DEVICES=0 \
python inference.py \
        -task ranking \
        -model roberta \
        -max_input 12800000 \
        -test queries=./data/queries.dev.small.tsv,docs=./data/collection.tsv,trec=./data/run.msmarco-passage.dev.small.trec \
        -vocab roberta-large \
        -pretrain roberta-large \
        -checkpoint ./checkpoints/roberta_large.bin \
        -res ./results/roberta-large_msmarco-dev.trec \
        -max_query_len 32 \
        -max_doc_len 221 \
        -batch_size 256
```

The checkpoints of roberta-large and electra-large are trained on MS-MARCO training data

```
wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz -P ./data
tar -zxvf ./data/triples.train.small.tar.gz -C ./data/ 
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

To train electra-large and roberta-large

First convert training data to jsonl version vis data_process.py

```python
import json
import codecs

def main():
    f_train_tsv = codecs.open('./data/triples.train.small.tsv','r')
    f_train_jsonl = codecs.open('./data/train.jsonl', 'w')
    cnt = 0
    for line in f_train_tsv:
        s = line.strip().split('\t')
        f_train_jsonl.write(json.dumps({'query':s[0],'doc_pos':s[1],'doc_neg':s[2]}) + '\n')
        cnt += 1
        if cnt > 3000000:
            break
    f_train_jsonl.close()
    f_train_tsv.close()
    print(cnt)

if __name__ == "__main__":
    main()
```

```
python3 data_process.py
```

 Train electra-large

```
CUDA_VISIBLE_DEVICES=0 \
python train.py\
        -task ranking \
        -model bert \
        -train ./data/train.jsonl \
        -max_input 3000000 \
        -save ./checkpoints/electra_large.bin \
        -dev queries=./data/queries.dev.small.tsv,docs=./data/collection.tsv,qrels=./data/qrels.dev.small.tsv,trec=./data/run.msmarco-passage.dev.small.100.trec \
        -qrels ./data/qrels.dev.small.tsv \
        -vocab google/electra-large-discriminator \
        -pretrain google/electra-large-discriminator \
        -res ./results/electra_large.trec \
        -metric mrr_cut_10 \
        -max_query_len 32 \
        -max_doc_len 256 \
        -epoch 1 \
        -batch_size 2 \
        -lr 5e-6 \
        -eval_every 10000
```

Train roberta-large

```
CUDA_VISIBLE_DEVICES=0 \
python train.py \
        -task ranking \
        -model roberta \
        -train ./data/train.jsonl \
        -max_input 3000000 \
        -save ./checkpoints/roberta_large.bin \
        -dev queries=./data/queries.dev.small.tsv,docs=./data/collection.tsv,qrels=./data/qrels.dev.small.tsv,trec=./data/run.msmarco-passage.dev.small.100.trec \
        -qrels ./data/qrels.dev.small.tsv \
        -vocab roberta-large \
        -pretrain roberta-large \
        -res ./results/roberta_large.trec \
        -metric mrr_cut_10 \
        -max_query_len 32 \
        -max_doc_len 256 \
        -epoch 1 \
        -batch_size 1 \
        -lr 5e-7 \
        -eval_every 20000
```



Since the whole dev dataset is too large, we only evaluate on top100 when training, and inference on whole dataset.
