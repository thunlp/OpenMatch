set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3
LR=2e-5

MAX_STEPS=80000
EPOCH=6

Q='full'

LOG_STEP=1000
EVAL_EVERY=1000

BATCH_SIZE=8
NEG=1


ckpt="/home/huxiaomeng/t5v11large/"

python -m torch.distributed.launch \
         --nproc_per_node=4 \
         --master_port=21227  \
        train.py \
        -task classification  \
        -model t5  \
        -qrels /data/private/yushi/collections/msmarco-passage/qrels.train.tsv     \
        -train $Q-q-$NEG-n_msmarco_train.jsonl  \
        -max_input 80000000  \
        -save /data/private/huxiaomeng/promptir/checkpoints/testt5v11/q$Q-n-$NEG/  \
        -dev dev/q500_msmarco_dev.jsonl   \
        -vocab $ckpt          \
        -pretrain $ckpt  \
        -res /data/private/huxiaomeng/promptir/results/testt5v11/q$Q-n-$NEG.trec  \
        -metric mrr_cut_10  \
        -max_query_len 76  \
        -max_doc_len 290  \
        -epoch $EPOCH  \
        -batch_size $BATCH_SIZE  \
        -lr $LR  \
        -eval_every $EVAL_EVERY  \
        -optimizer adamw  \
        -dev_eval_batch_size 128  \
        -n_warmup_steps 30  \
        -logging_step $LOG_STEP  \
        --log_dir=/data/private/huxiaomeng/promptir/logs/testt5v11/q$Q-n-$NEG/ \
        --max_steps=$MAX_STEPS \
        --t5v11=True
        
       


