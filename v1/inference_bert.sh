CUDA_VISIBLE_DEVICES=0 \
python inference.py \
        -task classification \
        -model bert \
        -max_input 1280000 \
        -test ./data/test_toy.jsonl \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -checkpoint ./checkpoints/bert.bin \
        -res ./results/bert.trec \
        -max_query_len 32 \
        -max_doc_len 256 \
        -batch_size 32
