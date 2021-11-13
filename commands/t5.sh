set -ex
export CUDA_VISIBLE_DEVICES=0,4
LR=2e-5

MAX_STEPS=80000
EPOCH=3

Q='full'
LOG_STEP=10
EVAL_EVERY=500

BATCH_SIZE=4
NEG=1


ckpt="t5-large"
#ckpt="t5-large"
python -m torch.distributed.launch \
         --nproc_per_node=2 \
         --master_port=2227  \
        train.py \
        -task classification  \
        -model t5  \
        -qrels /data/private/yushi/collections/msmarco-passage/qrels.train.tsv     \
        -train /data/private/huxiaomeng/promptir/dataset/msmarco/train/$Q-q-$NEG-n.jsonl \
        -dev /data/private/huxiaomeng/promptir/dataset/msmarco/dev/500-q.jsonl  \
        -max_input 80000000  \
        -save /data/private/huxiaomeng/test/checkpoints/q$Q-n-$NEG/  \
        -vocab $ckpt          \
        -pretrain $ckpt  \
        -res /data/private/huxiaomeng/test/results/q$Q-n-$NEG.trec  \
        -metric mrr_cut_10  \
        -max_query_len 76  \
        -max_doc_len 290  \
        -epoch $EPOCH  \
        -batch_size $BATCH_SIZE  \
        -lr $LR  \
        -eval_every $EVAL_EVERY  \
        -optimizer adafactor  \
        -dev_eval_batch_size 128  \
        -n_warmup_steps 0  \
        -logging_step $LOG_STEP  \
        --log_dir=/data/private/huxiaomeng/test/logs/q$Q-n-$NEG/ \
        --max_steps=$MAX_STEPS \
        -gradient_accumulation_steps 8 \
        --soft_prompt   \
        --soft_sentence="The following is a query-document pair. What we need to do is to find their relevance according to the probability that the predict word to be true"
        #--original_t5 \
        
       


