#!/bin/bash
## ************************************
## GPU
export gpu_num=4 ## GPU Number
export master_port=23900
export job_name=MetaBERT

## ************************************
export DATA_DIR= ## please set your dataset path here.
export SAVE_DIR= ## please set your saving path here.

## ************************************
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -u -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port $master_port meta_dist_train.py \
-job_name $job_name \
-save_folder $SAVE_DIR/results \
-model bert \
-task ranking \
-max_input 12800000 \
-train queries=$DATA_DIR/queries.train.tsv,docs=$DATA_DIR/collection.tsv,qrels=$DATA_DIR/qrels.train.tsv,trec=$DATA_DIR/trids_bm25_marco-10.tsv \
-dev queries=$DATA_DIR/queries.dev.small.tsv,docs=$DATA_DIR/collection.tsv,qrels=$DATA_DIR/qrels.dev.small.tsv,trec=$DATA_DIR/run.msmarco-passage.dev.small.100.trec \
-target trec=$DATA_DIR/devids_bm25_marco.tsv \
-qrels $DATA_DIR/qrels.dev.small.tsv \
-vocab bert-base-uncased \
-pretrain bert-base-uncased \
-metric mrr_cut_10 \
-max_query_len 32 \
-max_doc_len 221 \
-epoch 3 \
-train_batch_size 8 \
-target_batch_size 16 \
-gradient_accumulation_steps 2 \
-dev_eval_batch_size 1024 \
-lr 3e-6 \
-n_warmup_steps 160000 \
-logging_step 2000 \
-eval_every 10000 \
-eval_during_train \
