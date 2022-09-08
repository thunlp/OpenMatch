CUDA_VISIBLE_DEVICES=0 \
python train.py \
        -task ranking \
        -model cknrm \
        -train ./data/train_toy.jsonl \
        -max_input 1280000 \
        -save ./checkpoints/cknrm.bin \
        -dev ./data/dev_toy.jsonl \
        -qrels ./data/qrels_toy \
        -vocab ./data/glove.6B.300d.txt \
        -res ./results/cknrm.trec \
        -metric ndcg_cut_10 \
        -n_kernels 21 \
        -max_query_len 10 \
        -max_doc_len 150 \
        -epoch 2 \
        -batch_size 32 \
        -lr 1e-3 \
        -eval_every 10
