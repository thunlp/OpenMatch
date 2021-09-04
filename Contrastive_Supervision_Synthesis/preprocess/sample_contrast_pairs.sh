export dataset_name= ## you need to set this
export input_path= ## you need to set this
export generator_folder=qg_t5-base ## qg_t5-small ; qg_t5-base

python ./utils/sample_contrast_pairs.py \
--dataset_name $dataset_name \
--generator_folder $generator_folder \
--input_path $input_path \
--topk 100 \
--sample_n 5
