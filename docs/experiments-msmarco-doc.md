# MS MARCO Document Ranking
First, get the official data from [MSMARCO-Document-Ranking](https://github.com/microsoft/MSMARCO-Document-Ranking).

Get data and checkpoint from [Google Drive](https://drive.google.com/drive/folders/1cE_CUJFpfCUPOYSIDMYdz6_g3Zgccslj?usp=sharing). We provide the train data(qid \t pos\_did \t neg\_did), one can easily lookup query/doc texts from official files. The training data is generated from official train\_qrels and train\_top100 files, we randomly sampled 10 negative docs for each training query from top100 docs.

## Inference

### Full Ranking

### Re-Ranking

Preprocess dev dataset:
```
python data/preprocess.py -input_trec data/msmarco-docdev-top100 -input_queries data/msmarco-docdev-queries.tsv -input_docs data/msmarco-docs.tsv -output data/msmarco-doc.dev.jsonl
```

Reproduce BERT-Base-firstP, MRR@100(dev): 0.3590. *fiestP* mean all docs are truncated, only the first 512 sub-words are remained.

```
CUDA_VISIBLE_DEVICES=0 \
python inference.py \
        -task ranking \
        -model bert \
        -max_input 12800000 \
        -test ./data/msmarco-doc.dev.jsonl \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -checkpoint ./checkpoints/bert-base_marco-doc_firstp.bin \
        -res ./results/bert-base_msmarco-doc-dev_firstp.trec \
        -max_query_len 32 \
        -max_doc_len 477 \
        -batch_size 64
```

## train

Train.

```
CUDA_VISIBLE_DEVICES=0 \
python train.py \
        -task ranking \
        -model bert \
        -train  queries=./data/msmarco-doctrain-queries.tsv,docs=./data/msmarco-docs.tsv,qrels=./data/msmarco-doctrain-qrels.tsv,trec=./data/trids_marco-doc-10.tsv \
        -max_input 12800000 \
        -save ./checkpoints/bert.bin \
        -dev ./data/msmarco-doc.dev.jsonl \
        -qrels ./data/msmarco-docdev-qrels.tsv \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -res ./results/bert.trec \
        -metric mrr_cut_100 \
        -max_query_len 32 \
        -max_doc_len 477 \
        -epoch 3 \
        -batch_size 4 \
        -lr 3e-6 \
        -n_warmup_steps 100000 \
        -eval_every 10000
```
