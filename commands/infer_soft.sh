set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export OMP_NUM_THREADS=1
Model="bertprompt"
ckpt="roberta-large"
Q='full'
STEP=28000
NEG=1
Type="soft"
NEG_WORD=" irrelevant"
POS_WORD=" relevant"

python -m torch.distributed.launch \
         --nproc_per_node=4 \
         --master_port=16778  \
         ./dist_infer.py \
        -task prompt_classification \
        -model roberta \
        -max_input 8000000 \
        -vocab $ckpt \
        -pretrain $ckpt \
        -checkpoint /data/private/huxiaomeng/promptir/checkpoints/$Model/$Type/q$Q-n-$NEG/_step-$STEP.bin \
        -test /data/private/huxiaomeng/promptir/dataset/msmarco/test/all-q.jsonl \
        -res /data/private/huxiaomeng/promptir/results/$Model/$Type/test_q$Q-n-$NEG.trec \
        -max_query_len 83 \
        -max_doc_len 200 \
        -batch_size 100 \
        -neg_word " irrelevant" \
        -pos_word " relevant" \
        -template '[SP30] <q> <mask> <d>'  \
        --soft_prompt  \
