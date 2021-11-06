set -ex
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,2,3,4
ckpt="t5-large"
Q=5
STEP=250
NEG=1
Type=""
NEG_WORD="irrelevant"
POS_WORD="relevant"

python -m torch.distributed.launch \
         --nproc_per_node=4 \
         --master_port=16778  \
        OpenMatch-prompt/dist_infer.py \
        -task classification \
        -model t5 \
        -max_input 8000000 \
        -vocab $ckpt \
        -pretrain $ckpt \
        -checkpoint recover_simplet5/ckpt/q$Q-n-$NEG/_step-$STEP.bin \
        -test msmarco_test.jsonl \
        -res recover_simplet5/results/test_q$Q-n-$NEG.trec \
        -max_query_len 83 \
        -max_doc_len 290 \
        -batch_size 40  \
        -neg_word $NEG_WORD \
        -pos_word $POS_WORD \
        -template "<q> is [MASK] (relevant|irrevevant) to <d>"  \
