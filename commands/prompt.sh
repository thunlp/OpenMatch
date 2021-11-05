
set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
LR=5e-4
EPOCH=6

NEG_WORD=" irrelevant"
POS_WORD=" relevant"

MAX_STEPS=80000
Q=10000
NEG=1
LOG_STEP=50
EVAL_EVERY=50
BATCH_SIZE=8
MODEL="roberta"
ckpt="/data/private/yushi/pretrained_models/roberta-large"

TYPE="soft"
python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port=30009 \
train.py \
-task prompt_classification \
-model $MODEL \
-qrels /data/private/yushi/collections/msmarco-passage/qrels.train.tsv \
-train /data/private/huxiaomeng/promptir/dataset/msmarco/train/$Q-q-$NEG-n.jsonl \
-dev /data/private/huxiaomeng/promptir/dataset/msmarco/dev/500-q.jsonl  \
-max_input 80000000 \
-vocab $ckpt  \
-pretrain $ckpt  \
-metric mrr_cut_100  \
-max_query_len 76  \
-max_doc_len 290 \
-epoch $EPOCH  \
-batch_size $BATCH_SIZE  \
-lr $LR  \
-eval_every $EVAL_EVERY  \
-optimizer adamw   \
-dev_eval_batch_size 128  \
-n_warmup_steps 0  \
-logging_step $LOG_STEP  \
-save /data/private/huxiaomeng/promptir/checkpoints/bertprompt/$TYPE/q$Q-n-$NEG/ \
-res  /data/private/huxiaomeng/promptir/results/bertprompt/$TYPE/q$Q-n-$NEG.trec  \
--log_dir=/data/private/huxiaomeng/promptir/logs/bertprompt/$TYPE/q$Q-n-$NEG/  \
--max_steps=$MAX_STEPS  \
--pos_word=" relevant"  \
--neg_word=" irrelevant"  \
--template='[SP30] <q> <mask> <d>'  \
--soft_prompt  \


