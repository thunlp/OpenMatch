
set -ex
export CUDA_VISIBLE_DEVICES=0,4
export OMP_NUM_THREADS=1
LR=2e-5
EPOCH=300000

NEG_WORD=" irrelevant"
POS_WORD=" relevant"

MAX_STEPS=3000
Q=5
NEG=1
LOG_STEP=10
EVAL_EVERY=10
BATCH_SIZE=1
MODEL="roberta"
ckpt="/data/private/yushi/pretrained_models/roberta-large"

TYPE="test"
python -m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=3119 \
train.py \
-task prompt_classification \
-model $MODEL \
-qrels /data/private/yushi/collections/msmarco-passage/qrels.train.tsv \
-train /data/private/huxiaomeng/promptir/dataset/msmarco/train/$Q-q-$NEG-n.jsonl \
-dev /data/private/huxiaomeng/promptir/dataset/msmarco/dev/5-q.jsonl  \
-max_input 80000000 \
-vocab $ckpt  \
-pretrain $ckpt  \
-metric mrr_cut_10  \
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
-save /data/private/huxiaomeng/test/checkpoints/q$Q-n-$NEG/ \
-res  /data/private/huxiaomeng/test/results/q$Q-n-$NEG.trec  \
--log_dir=/data/private/huxiaomeng/test/logs/q$Q-n-$NEG/  \
--max_steps=$MAX_STEPS  \
--pos_word=" relevant"  \
--neg_word=" irrelevant"  \
--template='[SP30] <q> <mask> <d>'  \
--soft_prompt  \
-gradient_accumulation_steps 1


