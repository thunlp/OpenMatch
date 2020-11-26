CUDA_VISIBLE_DEVICES=0 \
python train.py \
        -task classification \
        -model bert \
        -train ./data/train_clas_toy.jsonl \
        -max_input 1280000 \
        -save ./checkpoints/bert.bin \
        -dev ./data/dev_toy.jsonl \
        -qrels ./data/qrels_toy \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -res ./results/bert.trec \
        -metric ndcg_cut_10 \
        -max_query_len 32 \
        -max_doc_len 256 \
        -epoch 1 \
        -batch_size 4 \
        -lr 2e-5 \
        -eval_every 10
