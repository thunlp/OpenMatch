## Introduction

We reproduced the work of [*Xiong et al - 2020 - Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text*](https://arxiv.org/pdf/2007.00808.pdf). The code is modified from its official repo [ANCE](https://github.com/microsoft/ANCE). There are some minor changes but not included the algorithms. In order to keep the same environment, we recommend to use the [Dockerfile](https://github.com/thunlp/OpenMatch/blob/master/Dockerfile) we provide.

## Data Preparation

The ANCE originally supports data format of MSMARCO (both passage and document). Several data preprocessing steps are still required before training.

### Data download

This commands will download both raw data required by both passage and document tasks. You can select the part you actually need. The data will be stored at  `$raw_data_dir=/path/to/OpenMatch/retrievers/ANCE/data/raw_data`  by default.

```shell
# assume you are currently located at ance folder
cd /path/to/OpenMatch/retrievers/ANCE/commands
bash data_download.sh
```

### Cleaning and processing data

The command to preprocess passage and document data is listed below, the processed data will be stored at `$preprocessed_data_dir` . Note that different types of tasks may require different lengths or processing methods, thus `$preprocessed_data_dir`  should  be named distinguishly.

```shell
cd /path/to/OpenMatch/retrievers/ANCE/data
python msmarco_data.py 
--data_dir $raw_data_dir \
--out_data_dir $preprocessed_data_dir \ 
--model_type {use rdot_nll for ANCE FirstP, rdot_nll_multi_chunk for ANCE MaxP} \ 
--model_name_or_path roberta-base \ 
--max_seq_length {use 512 for ANCE FirstP, 2048 for ANCE MaxP} \ 
--data_type {use 1 for passage, 0 for document}
```

Commands provided below are the parameters we used.

##### FirstP passage processing

```shell
cd /path/to/OpenMatch/retrievers/ANCE/data
python data/msmarco_data.py \
--data_dir data/raw_data \
--out_data_dir data/raw_data/ann_data_roberta-base_firstp_pas_512 \
--model_type rdot_nll \
--model_name_or_path roberta-base \
--max_seq_length 512 \
--data_type 1
```

##### FirstP document processing

```shell
cd /path/to/OpenMatch/retrievers/ANCE/data
python msmarco_data.py \
--data_dir ./raw_data \
--out_data_dir ./raw_data/ann_data_roberta-base_firstp_doc_512 \
--model_type rdot_nll \
--model_name_or_path roberta-base \
--max_seq_length 512 \
--data_type 0
```

##### MaxP document processing

```shell
cd /path/to/OpenMatch/retrievers/ANCE/data
python msmarco_data.py \
--data_dir ./raw_data \
--out_data_dir ./raw_data/ann_data_roberta-base_maxp_doc_2048 \
--model_type rdot_nll_multi_chunk \
--model_name_or_path roberta-base \
--max_seq_length 2048 \
--data_type 0 
```



## Training

### Warmup step

Before actual ANCE training, ANCE uses short passages to train the retriever with BM25 retrieval. Commands provided below are the parameters we used. The checkpoint will be saved at `/path/to/OpenMatch/retrievers/ANCE/data/msmarco_passage_warmup_checkpoints/`.

TODO: Checkpoint for warmup step.

```shell
cd /path/to/OpenMatch/retrievers/ANCE/commands
python3 ../drivers/run_warmup.py \
--train_model_type rdot_nll --model_name_or_path roberta-base \
--task_name MSMarco_passage_warmup --do_train --resume_train \
--data_dir ../data/raw_data \
--max_seq_length 128 \
--per_gpu_eval_batch_size=64 \
--per_gpu_train_batch_size=32 \
--learning_rate 2e-4  \
--logging_steps 1000   \
--num_train_epochs 2.0  \
--output_dir ../data/msmarco_passage_warmup_checkpoints/ \
--warmup_steps 1000  \
--save_steps 30000 \
--gradient_accumulation_steps 1 \
--expected_train_size 35000000 \
--logging_steps_per_eval 30 \
--fp16 \
--optimizer lamb \
--log_dir $tensorboard_log_path
```

### Inital embedding generation

This step will generate the initial embedding data with the warmup checkpoint, which is neccesary to ANCE training. In parameter  `--nproc_per_node=n` , `n` should be equal to the GPU number you use. For the GTX 1080Ti we used, the `--per_gpu_eval_batch_size` is set to 16.

General:

```shell
# embedding initialization
cd /path/to/OpenMatch/retrievers/ANCE/commands
python -m torch.distributed.launch --nproc_per_node={number of gpu} ../drivers/run_ann_data_gen.py \
--training_dir {path to save ann training checkpoints in the future} \
--init_model_dir {path to warmup checkpoint} \
--model_type {rdot_nll or rdot_nll_multi_chunk} --output_dir {path to save emebedding} \
--cache_dir {cache folder} \
--data_dir {preprocessed data} --max_seq_length {seq_len} \
--per_gpu_eval_batch_size {batch size up to GPU memory} --topk_training 200 --negative_sample 20 --end_output_num 0
```

Specifics:

```shell
# FirstP Passage initialization
cd /path/to/OpenMatch/retrievers/ANCE/commands
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 ../drivers/run_ann_data_gen.py \
--training_dir ../data/raw_data/FirstP_Pas512/ \
--init_model_dir ../data/msmarco_passage_warmup_checkpoints/checkpoint-420000/ \
--model_type rdot_nll --output_dir ../data/raw_data/FirstP_Pas512/ann_data/ \
--cache_dir ../data/raw_data/FirstP_Pas512/ann_data/cache/ \
--data_dir ../data/raw_data/ann_data_roberta-base_firstp_pas_512 --max_seq_length 512 \
--per_gpu_eval_batch_size 16 --topk_training 200 --negative_sample 20 --end_output_num 0

# FirstP Doc initialization
cd /path/to/OpenMatch/retrievers/ANCE/commands
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 ../drivers/run_ann_data_gen.py \
--training_dir ../data/raw_data/FirstP_Doc512/ \
--init_model_dir ../data/msmarco_passage_warmup_checkpoints/checkpoint-420000/ \
--model_type rdot_nll --output_dir ../data/raw_data/FirstP_Doc512/ann_data/ \
--cache_dir ../data/raw_data/FirstP_Doc512/ann_data/cache/ \
--data_dir ../data/raw_data/ann_data_roberta-base_firstp_doc_512 --max_seq_length 512 \
--per_gpu_eval_batch_size 16 --topk_training 200 --negative_sample 20 --end_output_num 0

# MaxP document initialization
cd /path/to/OpenMatch/retrievers/ANCE/commands
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 ../drivers/run_ann_data_gen.py --training_dir ../data/raw_data/MaxP_Doc2048/ \
--init_model_dir ../data/msmarco_passage_warmup_checkpoints/checkpoint-420000/ \
--model_type rdot_nll_multi_chunk --output_dir ../data/raw_data/MaxP_Doc2048/ann_data/ \
--cache_dir ../data/raw_data/MaxP_Doc2048/ann_data/cache/ \
--data_dir ../data/raw_data/ann_data_roberta-base_maxp_doc_2048 --max_seq_length 2048 \
--per_gpu_eval_batch_size 16 --topk_training 200 --negative_sample 20 --end_output_num 0
```



### ANN Training

We reproduced the training of FirstP-Passage, FirstP-Document and MaxP-Document in the paper. 

TODO: Release the checkpoints for these three tasks.

General:

```shell
python ../drivers/run_ann.py --model_type {rdot_nll or rdot_nll_multi_chunk} \
--model_name_or_path {path to warmup checkpoint} \
--task_name MSMarco --triplet --data_dir {preprocessed data} \
--ann_dir {path to save emebedding} --max_seq_length {seq_len} --per_gpu_train_batch_size=2 \
--gradient_accumulation_steps 2 --learning_rate 1e-6 --output_dir {path to save ann training checkpoints} \
--warmup_steps 5000 --logging_steps 100 --save_steps 10000 --optimizer lamb --single_warmup \
--log_dir {tensorboard folder}
```

Specifics:

```shell
# FirstP Passage 
cd /path/to/OpenMatch/retrievers/ANCE/commands
python ../drivers/run_ann.py --model_type rdot_nll \
--model_name_or_path ../data/msmarco_passage_warmup_checkpoints/checkpoint-420000/ \
--task_name MSMarco --triplet --data_dir ../data/raw_data/ann_data_roberta-base_firstp_pas_512 \
--ann_dir ../data/raw_data/FirstP_Pas512/ann_data/ --max_seq_length 512 --per_gpu_train_batch_size=2 \
--gradient_accumulation_steps 2 --learning_rate 1e-6 --output_dir ../data/raw_data/FirstP_Pas512/ \
--warmup_steps 5000 --logging_steps 100 --save_steps 10000 --optimizer lamb --single_warmup \
--log_dir {tensorboard folder}

# FirstP Doc 
cd /path/to/OpenMatch/retrievers/ANCE/commands
python ../drivers/run_ann.py --model_type rdot_nll \
--model_name_or_path ../data/msmarco_passage_warmup_checkpoints/checkpoint-420000/ \
--task_name MSMarco --triplet --data_dir ../data/raw_data/ann_data_roberta-base_firstp_doc_512/ \
--ann_dir ../data/raw_data/FirstP_Doc512/ann_data/ --max_seq_length 512 --per_gpu_train_batch_size=2 \
--gradient_accumulation_steps 2 --learning_rate 5e-6 --output_dir ../data/raw_data/FirstP_Doc512/ \
--warmup_steps 3000 --logging_steps 100 --save_steps 10000 --optimizer lamb --single_warmup \
--log_dir {tensorboard folder}

# MaxP document
cd /path/to/OpenMatch/retrievers/ANCE/commands
python ../drivers/run_ann.py --model_type rdot_nll_multi_chunk  \
--model_name_or_path ../data/msmarco_passage_warmup_checkpoints/checkpoint-420000/ \
--task_name MSMarco --triplet --data_dir  ../data/raw_data/ann_data_roberta-base_maxp_doc_2048 \
--ann_dir ../data/raw_data/MaxP_Doc2048/ann_data/ --max_seq_length 2048 --per_gpu_train_batch_size=1 \
--gradient_accumulation_steps 16 --learning_rate 1e-5 --output_dir ../data/raw_data/MaxP_Doc2048/ \
--warmup_steps 500 --logging_steps 100 --save_steps 10000 --optimizer lamb --single_warmup \
--log_dir {tensorboard folder}
```



Parallel embedding generation should run with the training process. The command is basically the same as the intialization generation.

```shell
cd /path/to/OpenMatch/retrievers/ANCE/commands
python -m torch.distributed.launch --nproc_per_node={number of gpu} ../drivers/run_ann_data_gen.py \
--training_dir {path to save ann training checkpoints in the future} \
--init_model_dir {path to warmup checkpoint} \
--model_type {rdot_nll or rdot_nll_multi_chunk} --output_dir {path to save emebedding} \
--cache_dir {cache folder} \
--data_dir {preprocessed data} --max_seq_length {seq_len} \
--per_gpu_eval_batch_size {batch size up to GPU memory} --topk_training 200 --negative_sample 20
```



## Evaluation

We modified the evaluation code of the original one in order to save an extra trec file for the retrieval results. To do the ANN retrieval evaluation and see the performance of a checkpoint, you will need **both the checkpoint file and the corresponding embeddings**. 

Firstly, modify the parameters in `/path/to/OpenMatch/retrievers/ANCE/evaluation/Calculate_Metrics.py`

```python
# an example for FirstP Passage evaluation
task_name= 'FirstP_Pas_512' # custom name
checkpoint_path = "../data/raw_data/FirstP_Pas512/ann_data/" # the folder contains all the checkpoint folders
checkpoint = 520000 # specific checkpoint step
data_type = 1 # 0 for document, 1 for passage
test_set = 0 # 0 for dev_set(passage), 1 for eval_set(document)
raw_data_dir = "../data/raw_data/"
processed_data_dir = "../data/raw_data/ann_data_roberta-base_firstp_pas_512/" 
```

Run the command:

```shell
cd /path/to/OpenMatch/retrievers/ANCE/evaluation
python Calculate_Metrics.py
```



Note that original ANCE code will use relative query and document ids, which are calculated during data pre-processing step. If you need the trec file with qids&pids in MSMARCO or whatever customed dataset, you need to run the following:

```shell
cd /path/to/OpenMatch/retrievers/ANCE/evaluation
python convert_trec.py
```

Parameters

```python
data_type = 1 # 0 for document, 1 for passage
test_set = 0 # 0 for dev_set(passage), 1 for eval_set(document)
processed_data_dir = "../data/raw_data/ann_data_roberta-base_firstp_pas_512/" 
```

The converted trec file will be saved with postfix `.formatted.trec`.

## Inference

The command for inference embedding generation is the same as parallel embedding generation while training. After the generation, you can refer to the *Evaluation* section above to get the performance result.

```shell
cd /path/to/OpenMatch/retrievers/ANCE/commands
python -m torch.distributed.launch --nproc_per_node={number of gpu} ../drivers/run_ann_data_gen.py \
--training_dir {path to save ann training checkpoints in the future} \
--init_model_dir {path to warmup checkpoint} \
--model_type {rdot_nll or rdot_nll_multi_chunk} --output_dir {path to save emebedding} \
--cache_dir {cache folder} \
--data_dir {preprocessed data} --max_seq_length {seq_len} \
--per_gpu_eval_batch_size {batch size up to GPU memory} --topk_training 200 --negative_sample 20
```

## Data Format

The data format of ANCE is the same as MSMARCO, official introduction [here](https://github.com/microsoft/MSMARCO-Passage-Ranking#data-information-and-formating).

Specifically, taking document data format as example,  it requires the following data files for each task:

- Document file: `{document_id} {raw text}`

- Query training file: `{query_id} {raw text}`

- Qrels training file: `{query_id} {dummy_tag} {document_id} {relevancy, 0 for not relevant}`

- Query evaluation file: `{query_id} {raw text}`

  

The description behind is the format of each line of the content. You can use your own dataset with such format to train ANCE.


