CUDA_VISIBLE_DEVICES=0 \
python gen_feature.py \
        -task classification \
        -model bert \
        -max_input 1280000 \
        -dev ./data/dev_toy.jsonl \
        -vocab bert-base-uncased \
        -pretrain bert-base-uncased \
        -checkpoint ./checkpoints/bert.bin \
        -res ./features/bert_features \
        -max_query_len 32 \
        -max_doc_len 256 \
        -batch_size 32
