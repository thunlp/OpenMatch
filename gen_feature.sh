CUDA_VISIBLE_DEVICES=0 \
python gen_feature.py \
        -task ranking \
        -model cknrm \
        -max_input 1280000 \
        -vocab ./data/glove.6B.300d.txt \
        -checkpoint ./checkpoints/cknrm.bin \
        -dev ./data/dev_toy.jsonl \
        -res ./features/cknrm.trec \
        -max_query_len 10 \
        -max_doc_len 256 \
        -batch_size 32
