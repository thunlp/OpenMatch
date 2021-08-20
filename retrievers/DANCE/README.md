# More Robust Dense Retrieval with Contrastive Dual Learning


This repository provides the implementation code for the paper DANCE--[More Robust Dense Retrieval with Contrastive Dual Learning](http://arxiv.org/abs/2107.07773). 

Our code inherits from the repository [ANCE](https://github.com/microsoft/ANCE). While most of the settings should be the same as the original ANCE implementation, we add some code to conduct faster ANN search with GPUs, and to search through splitted shards of document embeddings for memory-saving purpose.


## Environment

```shell
git clone https://github.com/thunlp/DANCE.git
cd DANCE
python ANCE_setup.py install
conda install --yes --file requirements.txt
```

Note that [apex](https://github.com/NVIDIA/apex) is required if you want to train with half-precision.

## Data Processing

### Download

DANCE is trained on the MS MARCO Document dataset.

```shell
bash commands/datadownload.sh
```

### Pre-processing

```shell
cd data
python msmarco_data.py \
--data_dir ./raw_data \
--out_data_dir [processed data folder] \
--model_type rdot_nll \
--model_name_or_path roberta-base \
--max_seq_length 512 \
--data_type 0 
```

Since we split the devset into 2 folds during training to select checkpoint, we additionally save a dictionary to keep the split information. What's more, we want to test the model performance on the dual task, so the reversed format of the dev qrels file in the main task is also produced in our code.

```shell
python validation_split.py ./raw_data/msmarco-docdev-qrels.tsv 2
```


## Training

The training requires two parallel running processes:
- training process: update the model parameters with the training batch feed.

- inference process: use the latest checkpoint saved by the training process to produce new embeddings and asynchronously update the document index.


DANCE is trained with two seperate stages, which is descriped as the following.

### Normalization Tuning

The DANCE training continues from the ANCE released checkpoint. The `[ANCE checkpoint folder]` should be prepared before training. Before training, we need to run the inference process and get the embeddings for the all the queries and documents. The **inference** proecess runs with 4 GeForce RTX 2080 Ti GPUs of 11GB.

```shell
python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=57135 ../drivers/run_ann_data_gen.py \
--representation_l2_normalization \
--d2q_task_evaluation --d2q_task_marco_dev_qrels ../data/raw_data/msmarco-docdev-qrels-d2q-reversed.tsv \
--model_type rdot_nll --data_type 0 --dev_split_num 2 \
--split_ann_search --gpu_index \
--data_dir [processed data folder] \
--init_model_dir [ANCE checkpoint folder] \
--training_dir [normalization tuning task folder] \
--output_dir  [normalization tuning task folder]/ann_data/ \
--cache_dir  [normalization tuning task folder]/ann_data/cache/ \
--max_seq_length 512 --per_gpu_eval_batch_size 210 \
--topk_training 200 --negative_sample 20 --end_output_num -1 
```

After the inference process finishing the inital inference with the ANCE checkpoint, we can start the Normalization **training** proecess runs with 8 GeForce RTX 2080 Ti GPUs of 11GB.

```shell
python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=47134 ../drivers/run_ann.py \
--model_name_or_path [ANCE checkpoint folder] \
--loss_objective_function "simclr_cosine" --temperature 0.01 --representation_l2_normalization \
--model_type rdot_nll --fp16 \
--task_name [taskname] \
--triplet --data_dir [processed data folder] \
--output_dir [normalization tuning task folder] \
--ann_dir [normalization tuning task folder]/ann_data/ \
--max_seq_length 512 --per_gpu_train_batch_size=4 \
--gradient_accumulation_steps 2 --learning_rate 5e-6 \
--warmup_steps 3000 --logging_steps 100 --save_steps 5000 --optimizer lamb --single_warmup \
--log_dir ../log/tensorboard/msmarco_document_firstP_ann_train/logs/[taskname]
```

During training, the checkpoints will be saved at the path `[Normalization Tuning task folder]`. 

### Dual Training

After the first stage, the best-perform checkpoint  `[best normalization stage checkpoint folder]` will be selected. In the second training stage, the dual training are introduced and continue with this checkpoint. The settings are similar with those in the first stage. An initial **inference** is still required. Parameter `--dual_training` is used to activate the dual training.

```shell
python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=57135 ../drivers/run_ann_data_gen.py \
--dual_training --representation_l2_normalization \
--d2q_task_evaluation --d2q_task_marco_dev_qrels ../data/raw_data/msmarco-docdev-qrels-d2q-reversed.tsv \
--model_type rdot_nll --data_type 0 --dev_split_num 2 \
--split_ann_search --gpu_index \
--data_dir [processed data folder] \
--init_model_dir [best normalization stage checkpoint folder] \
--training_dir [dual training task folder] \
--output_dir [dual training task folder]/ann_data/ \
--cache_dir [dual training task folder]/ann_data/cache/ \
--max_seq_length 512 --per_gpu_eval_batch_size 210 \
--topk_training 200 --negative_sample 20 --end_output_num -1 
```

Notice that we use a distinct folder `[dual training task folder]` to store the training data in the second stage.

The command for **training** process:


```shell
python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=47134 ../drivers/run_ann.py \
--model_name_or_path [best normalization stage checkpoint folder] \
--dual_training --dual_loss_weight=0.1 \
--loss_objective_function "simclr_cosine" --temperature 0.01 --representation_l2_normalization \
--model_type rdot_nll --fp16 \
--task_name [taskname] \
--triplet --data_dir [processed data folder] \
--output_dir [dual training task folder] \
--ann_dir [dual training task folder]/ann_data/ \
--max_seq_length 512 --per_gpu_train_batch_size=4 \
--gradient_accumulation_steps 2 --learning_rate 5e-6 \
--warmup_steps 3000 --logging_steps 100 --save_steps 5000 --optimizer lamb --single_warmup \
--log_dir ../log/tensorboard/msmarco_document_firstP_ann_train/logs/[taskname]
```

### Extra Options

Different from the original ANCE implementation, we provide extra options in the inference code:
- `--gpu_index ` for indexing and searching with GPU, which is faster than CPU in our case.
- `--split_ann_search ` for conducting ANN search through splitted embedding shards, which will largely save the memory usage since it will only load part of the embeddings at each time. Note that the number of split shards are still **the same as number of GPUs used in inference process**.

## Independent Inference and Test

We provide the code to independently inference with a given checkpoint:

```shell
python -u -m torch.distributed.launch \
--nproc_per_node=[num GPU] --master_port=57135 ../drivers/run_ann_data_inference_eval.py \
--model_type rdot_nll --data_type 0 --dev_split_num 2 \
--split_ann_search --gpu_index \
--init_model_dir [checkpoint folder] \
--inference_one_specified_ckpt \
--data_dir [processed data folder] \
--training_dir [training task folder] \
--output_dir [training task folder]/ann_data/ \
--cache_dir [training task folder]/ann_data/cache/ \
--max_query_length 64 --max_seq_length 512 --per_gpu_eval_batch_size 256
```

The code for evaluation are in `evaluation/Calculate_Metrics.py`.

## Inference for Custom MSMARCO-like Dataset 

We provide codes for MSMARCO-like datasets.

In such dataset, several files must be provided:
- documents
    - `[document collection.tsv]`: each line contains `[passage id]\t[text]\n` for the document texts. `[passage id]`s are in format "Dxxxxx", where "xxxxx" are integers.
- files need be provided for each query set. (training, dev, eval, etc.)
    - `[custom queries.tsv]`: `[query id]\t[text]\n` for lines. `[query id]` is also integers.
    - `[custom qrels.tsv]`: `[query id] 0 [passage id] 1\n` for lines. This is optional because we may not have answers for the testset queries.

Pre-processing:

```shell
python data/custom_data.py \
--data_dir [raw tsv data folder] \
--out_data_dir [processed data folder] \
--model_type rdot_nll \
--model_name_or_path roberta-base \
--max_seq_length 512 \
--data_type 0 \
--doc_collection_tsv [doc text path] \
--save_prefix [query saving name] \
--query_collection_tsv [query text path] \
--qrel_tsv [optional qrel tsv] \
```

You can specify a pytorch checkpoint and use it to inference the embeddings of the documents or queries.

```shell
python -u -m torch.distributed.launch \
--nproc_per_node=[num GPU] --master_port=57135 ./drivers/run_ann_emb_inference.py \
--model_type rdot_nll \
--inference_type query --save_prefix [prefix of the query preprocessed file. eg., train] \
--split_ann_search --gpu_index \
--init_model_dir [checkpoint folder] \
--data_dir [processed data folder] \
--training_dir [task folder] \
--output_dir [task folder]/ann_data/ \
--cache_dir [task folder]/ann_data/cache/ \
--max_query_length 64 --max_seq_length 512 --per_gpu_eval_batch_size 256
```

With using parameters `--inference_type query --save_prefix [prefix of the query preprocessed file. eg., train] \`, you can inference different sets of queries. 
With using parameters `--inference_type document` and removing ` --save_prefix`, you can inference the document embeddings. 

Next, you can use the following code to produce the trec format retrieval results of different query sets. Note that the embedding files will be matched by `emb_file_pattern = os.path.join(emb_dir,f'{emb_prefix}{checkpoint_postfix}__emb_p__data_obj_*.pb')`, check out how your embeddings are saved and specify the `checkpoint_postfix` for the program to load the embeddings.

```shell
python ./evaluation/retrieval.py \
--trec_path [output trec path] \
--emb_dir [folder dumpped query and passage/document embeddings which is output_dir, same as --output_dir above] \
--checkpoint_postfix [checkpoint custom name] \
--processed_data_dir [processed data folder] ] \
--queryset_prefix [query saving name] \
--gpu_index True --topN 100 --data_type 0 
```

Now you can play with the trec files and calculate different metrics .


## Checkpoints

TODO: upload the checkpoints.

| Checkpoint       | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| [DANCE](https://thunlp.oss-cn-qingdao.aliyuncs.com/OpenMatch/DANCE/DANCE.zip)        | Our final model fine-tuned with both Dual training and L2 normalization. |
| [ANCE w. Norm](https://thunlp.oss-cn-qingdao.aliyuncs.com/OpenMatch/DANCE/ANCE_w_Norm.zip) | Model fine-tuned with only L2 normalization.                 |
| [ANCE w. Dual](https://thunlp.oss-cn-qingdao.aliyuncs.com/OpenMatch/DANCE/ANCE_w_Dual.zip) | Model fine-tuned with only dual training.                    |
| [ANCE](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Document_ANCE_FirstP_Checkpoint.zip)         | Official released checkpoint for MS MARCO Document from the ANCE repo. |

