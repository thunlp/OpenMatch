gpu_no=8

# model type
model_type="dpr"
seq_length=256
triplet="--triplet --optimizer lamb" # set this to empty for non triplet model

# hyper parameters
batch_size=16
gradient_accumulation_steps=1
learning_rate=1e-5
warmup_steps=1000

# input/output directories
base_data_dir="../data/QA_NQ_data/" 
job_name="ann_NQ_test"
model_dir="${base_data_dir}${job_name}/"
model_ann_data_dir="${model_dir}ann_data/"
pretrained_checkpoint_dir="../../../DPR/checkpoint/retriever/multiset/bert-base-encoder.cp"

train_cmd="\
sudo python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_dpr.py --model_type $model_type \
--model_name_or_path $pretrained_checkpoint_dir --task_name MSMarco $triplet --data_dir $base_data_dir \
--ann_dir $model_ann_data_dir --max_seq_length $seq_length --per_gpu_train_batch_size=$batch_size \
--gradient_accumulation_steps $gradient_accumulation_steps --learning_rate $learning_rate --output_dir $model_dir \
--warmup_steps $warmup_steps --logging_steps 100 --save_steps 1000 --log_dir "~/tensorboard/${DLWS_JOB_ID}/logs/${job_name}" \
"

echo $train_cmd
eval $train_cmd

echo "copy current script to model directory"
sudo cp $0 $model_dir