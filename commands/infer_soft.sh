set -ex
export CUDA_VISIBLE_DEVICES=0,3
Model="bertprompt"
ckpt="roberta-large"
Q=5
STEP=500
NEG=1
Type="soft"
NEG_WORD=" irrelevant"
POS_WORD=" relevant"

python ../dist_infer.py \
        -task prompt_classification \
        -model roberta \
        -max_input 8000000 \
        -vocab $ckpt \
        -pretrain $ckpt \
        -checkpoint /data/private/huxiaomeng/promptir/checkpoints/$MODEL/$Type/q$Q-n-$NEG/_step-$STEP.bin \
        -test /data/private/huxiaomeng/promptir/dataset/msmarco/test/all-q.jsonl \
        -res /data/private/huxiaomeng/promptir/results/$MODEL/$Type/test_q$Q-n-$NEG.trec \
        -max_query_len 83 \
        -max_doc_len 200 \
        -batch_size 50 \
        -neg_word " irrelevant" \
        -pos_word " relevant" \
        --template='[SP30] <q> <mask> <d>'  \
        --soft_prompt  \
