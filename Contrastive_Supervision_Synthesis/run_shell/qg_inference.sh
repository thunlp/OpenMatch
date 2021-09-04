# !/bin/bash
## --------------------------------------------
export CUDA=2
export pretrain_generator_type=t5-base ## t5-small ; t5-base
export per_gpu_gen_batch_size=200 ## 200; 400
export target_dataset_name= ## you need to set this
export generator_mode=qg
## --------------------------------------------

## --------------------------------------------
export generator_load_dir= ## you need to set this
export target_dataset_dir= ## you need to set this
## --------------------------------------------

CUDA_VISIBLE_DEVICES=$CUDA python ../scripts/inference.py \
--generator_mode $generator_mode \
--pretrain_generator_type $pretrain_generator_type \
--per_gpu_gen_batch_size $per_gpu_gen_batch_size \
--generator_load_dir $generator_load_dir \
--target_dataset_dir $target_dataset_dir \
