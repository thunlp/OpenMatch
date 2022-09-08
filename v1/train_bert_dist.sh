CUDA_VISIBLE_DEVICES=0,1,2,3 \ # set visible CUDA GPUs
python -u -m torch.distributed.launch \ #lauch distributed training
--nproc_per_node=4 \ # number equals to how many GPUs used
--master_port=12345 train.py \
        -task ranking \
        -model bert \
        # do not use the single json file
        -train queries=/path/to/queries.tsv,docs=/path/to/docs.tsv,qrels=/path/to/qrels.tsv,trec=/path/to/trec.tsv \
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
        -eval_every 100 \
        -optimizer adamw \
        -dev_eval_batch_size 128 \
        -gradient_accumulation_steps 4 \
        -n_warmup_steps 10000 \
        -logging_step 100
