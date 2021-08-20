# # Passage ANCE(FirstP) 
gpu_no=4
seq_length=512
model_type=rdot_nll
tokenizer_type="roberta-base"
base_data_dir="../data/raw_data/"
preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}_dev/"
job_name="OSPass512"
pretrained_checkpoint_dir=""

# # Document ANCE(FirstP) 
# gpu_no=4
# seq_length=512
# model_type=rdot_nll
# tokenizer_type="roberta-base"
# base_data_dir="../data/raw_data/"
# preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
# job_name="OSDoc512"
# pretrained_checkpoint_dir=""

# # Document ANCE(MaxP)
# gpu_no=4
# seq_length=2048
# model_type=rdot_nll_multi_chunk
# tokenizer_type="roberta-base"
# base_data_dir="../data/raw_data/"
# preprocessed_data_dir="${base_data_dir}ann_data_${tokenizer_type}_${seq_length}/"
# job_name="OSDoc2048"
# pretrained_checkpoint_dir=""

##################################### Inference ################################
model_dir="${base_data_dir}${job_name}/"
model_ann_data_dir="${model_dir}ann_data_inf/"

initial_data_gen_cmd="\
python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen.py --training_dir $pretrained_checkpoint_dir \
--init_model_dir $pretrained_checkpoint_dir --model_type $model_type --output_dir $model_ann_data_dir \
--cache_dir "${model_ann_data_dir}cache/" --data_dir $preprocessed_data_dir --max_seq_length $seq_length \
--per_gpu_eval_batch_size 16 --topk_training 200 --negative_sample 20 --end_output_num 0 --inference \
"

echo $initial_data_gen_cmd
eval $initial_data_gen_cmd
