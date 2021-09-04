export dataset_name= ## you need to set this
export input_path=## you need to set this
export output_path=## you need to set this

python ./utils/prepro_dataset.py \
--dataset_name $dataset_name \
--input_path $input_path \
--output_path $output_path \
