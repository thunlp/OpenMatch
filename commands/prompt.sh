
set -ex
export CUDA_VISIBLE_DEVICES=5,6
LR=5e-4
EPOCH=3

NEG_WORD=" irrelevant"
POS_WORD=" relevant"

MAX_STEPS=80000
Q=1000
NEG=1
LOG_STEP=10
EVAL_EVERY=10
BATCH_SIZE=4
MODEL="roberta"
ckpt="/data/private/yushi/pretrained_models/roberta-base"

TYPE="soft"
python -m torch.distributed.launch --nproc_per_node=2 --master_port=40009 train.py -task prompt_classification -model $MODEL -qrels /data/private/yushi/collections/msmarco-passage/qrels.train.tsv -train /data/private/huxiaomeng/promptir/dataset/msmarco/train/$Q-q-$NEG-n.jsonl -max_input 80000000 -save /mnt/121_data/private/yushi121/checkpoints/bertprompt/$TYPE/q$Q-n-$NEG/ -dev /data/private/huxiaomeng/promptir/dataset/msmarco/dev/500-q.jsonl  -vocab $ckpt  -pretrain $ckpt  -res results/bertprompt/$TYPE/q$Q-n-$NEG.trec  -metric mrr_cut_10  -max_query_len 76  -max_doc_len 290 -epoch $EPOCH  -batch_size $BATCH_SIZE  -lr $LR  -eval_every $EVAL_EVERY  -optimizer adamw   -dev_eval_batch_size 128  -n_warmup_steps 100  -logging_step $LOG_STEP  --log_dir=logs/bertprompt/$TYPE/q$Q-n-$NEG/  --max_steps=$MAX_STEPS  --pos_word=" relevant"  --neg_word=" irrelevant"  --template='[SP30] <q> <mask> <d>'  --soft_prompt  -gradient_accumulation_steps 2

