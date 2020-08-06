# TREC COVID

## Inference
Get checkpoint.
* [checkpoints](https://drive.google.com/drive/folders/1YHCMdSI7clFqPdfrRHA786PIecIxtKqA?usp=sharing)

Get data from Google Drive.
* [round1](https://drive.google.com/open?id=17CEoLecus232pCDwCECaJD4vNfh4OQao)
* [round2](https://drive.google.com/open?id=1O6e8gXFnykkhN2icMCuWlMZkKUv6B3fV)

Filter round1 data from round2 data.
```
python data/filter.py \
  -input_qrels qrels-cord19-round1.txt \
  -input_trec anserini.covid-r2.fusion2.txt \
  -output_topk 50 \
  -output_trec anserini.covid-r2.fusion2-filtered.txt
```

Preprocess round2 data.
```
python ./data/preprocess.py \
  -input_trec anserini.covid-r2.fusion2-filtered.txt \
  -input_queries questions_cord19-rnd2.txt \
  -input_docs cord19_0501_titabs.jsonl \
  -output ./data/test_trec-covid-round2.jsonl
```

Reproduce scibert.
```
CUDA_VISIBLE_DEVICES=0 \
python inference.py \
        -task classification \
        -model bert \
        -max_input 1280000 \
        -test ./data/test_trec-covid-round2.jsonl \
        -vocab allenai/scibert_scivocab_uncased \
        -pretrain allenai/scibert_scivocab_uncased \
        -checkpoint ./checkpoints/scibert.bin \
        -res ./results/scibert.trec \
        -mode cls \
        -max_query_len 32 \
        -max_doc_len 256 \
        -batch_size 32
```

Reproduce reinfoselect scibert.
```
CUDA_VISIBLE_DEVICES=0 \
python inference.py \
        -task classification \
        -model bert \
        -max_input 1280000 \
        -test ./data/test_trec-covid-round2.jsonl \
        -vocab allenai/scibert_scivocab_uncased \
        -pretrain allenai/scibert_scivocab_uncased \
        -checkpoint ./checkpoints/reinfoselect_scibert.bin \
        -res ./results/reinfoselect_scibert.trec \
        -mode pooling \
        -max_query_len 32 \
        -max_doc_len 256 \
        -batch_size 32
```

## train
Get training data from [google drive](https://drive.google.com/file/d/1BT5gCOb1Kxkfh0BWqgUSgkxp2JPpRIWm/view?usp=sharing).

Train.
```
CUDA_VISIBLE_DEVICES=0 \
python train.py \
        -task classification \
        -model bert \
        -train ./data/seanmed.train.320K-pairs.jsonl \
        -max_input 1280000 \
        -save ./checkpoints/scibert.bin \
        -dev ./data/dev_trec-covid-round1.jsonl \
        -qrels qrels-cord19-round1.txt \
        -vocab allenai/scibert_scivocab_uncased \
        -pretrain allenai/scibert_scivocab_uncased \
        -res ./results/scibert.trec \
        -metric ndcg_cut_10 \
        -max_query_len 32 \
        -max_doc_len 256 \
        -epoch 5 \
        -batch_size 16 \
        -lr 2e-5 \
        -n_warmup_steps 4000 \
        -eval_every 200
```

For ReInfoSelect training:
```
CUDA_VISIBLE_DEVICES=0 \
python train.py \
        -task ranking \
        -model bert \
        -reinfoselect \
        -train ./data/seanmed.train.320K-pairs.jsonl \
        -max_input 1280000 \
        -save ./checkpoints/scibert_rl.bin \
        -dev ./data/dev_trec-covid-round1.jsonl \
        -qrels qrels-cord19-round1.txt \
        -vocab allenai/scibert_scivocab_uncased \
        -pretrain allenai/scibert_scivocab_uncased \
        -checkpoint ./checkpoints/scibert.bin \
        -res ./results/scibert_rl.trec \
        -metric ndcg_cut_10 \
        -max_query_len 32 \
        -max_doc_len 256 \
        -epoch 5 \
        -batch_size 16 \
        -lr 2e-5 \
        -tau 1 \
        -n_warmup_steps 1000 \
        -eval_every 1
```
