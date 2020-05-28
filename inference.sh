CUDA_VISIBLE_DEVICES=0 \
python inference.py \
        -task ranking \
        -model cknrm \
        -max_input 1280000 \
        -vocab ./data/glove.6B.300d.txt \
        -checkpoint ./checkpoints/cknrm.bin \
        -test ./data/test_toy.jsonl \
        -res ./results/cknrm.trec \
        -max_query_len 10 \
        -max_doc_len 256 \
        -batch_size 32
