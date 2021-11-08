set -ex
export CUDA_VISIBLE_DEVICES=0
LR=2e-5

MAX_STEPS=3000
EPOCH=30000

Q=50

LOG_STEP=10
EVAL_EVERY=50

BATCH_SIZE=4
NEG=1


ckpt="/data/private/yushi/pretrained_models/t5-large"

python -m torch.distributed.launch \
         --nproc_per_node=1 \
         --master_port=10227  \
        train.py \
        -task classification  \
        -model t5  \
        -qrels /data/private/yushi/collections/msmarco-passage/qrels.train.tsv     \
        -train /data/private/huxiaomeng/promptir/dataset/msmarco/train/$Q-q-$NEG-n.jsonl \
        -dev /data/private/huxiaomeng/promptir/dataset/msmarco/dev/5-q.jsonl  \
        -max_input 80000000  \
        -save /data/private/huxiaomeng/last_try_t5v11/ckpt/q$Q-n-$NEG/  \
        -vocab $ckpt          \
        -pretrain $ckpt  \
        -res /data/private/huxiaomeng/last_try_t5v11/results/q$Q-n-$NEG.trec  \
        -metric mrr_cut_10  \
        -max_query_len 76  \
        -max_doc_len 290  \
        -epoch $EPOCH  \
        -batch_size $BATCH_SIZE  \
        -lr $LR  \
        -eval_every $EVAL_EVERY  \
        -optimizer adafactor  \
        -dev_eval_batch_size 128  \
        -n_warmup_steps 30  \
        -logging_step $LOG_STEP  \
        --log_dir=/data/private/huxiaomeng/last_try_t5v11/logs/q$Q-n-$NEG/ \
        --max_steps=$MAX_STEPS \
        -gradient_accumulation_steps 1 \
        --original_t5 \
        
       


