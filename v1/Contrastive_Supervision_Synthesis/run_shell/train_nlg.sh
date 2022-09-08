# #!/bin/bash
export CUDA=0
export generator_mode=qg # qg / contrastqg
export pretrain_generator_type=t5-small ## t5-small / t5-base

export pretrain_model_dir=../data/pretrain_model
export train_file=../data/source_data/toy_triples.train.small.tsv
export save_dir=../results

## ------------------------------------------------------------------
## ------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$CUDA python ../scripts/train.py --run_mode train \
--generator_mode $generator_mode \
--pretrain_generator_type $pretrain_generator_type \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 1 \
--pretrain_model_dir $pretrain_model_dir \
--train_file $train_file \
--save_dir $save_dir \
