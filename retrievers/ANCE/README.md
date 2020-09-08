# Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval
Lee Xiong*, Chenyan Xiong*, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, Arnold Overwijk

This repo provides the code for reproducing the experiments in [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://arxiv.org/pdf/2007.00808.pdf) 

Conducting text retrieval in a dense learned representation space has many intriguing advantages over sparse retrieval. Yet the effectiveness of dense retrieval (DR)
often requires combination with sparse retrieval. In this paper, we identify that
the main bottleneck is in the training mechanisms, where the negative instances
used in training are not representative of the irrelevant documents in testing. This
paper presents Approximate nearest neighbor Negative Contrastive Estimation
(ANCE), a training mechanism that constructs negatives from an Approximate
Nearest Neighbor (ANN) index of the corpus, which is parallelly updated with the
learning process to select more realistic negative training instances. This fundamentally resolves the discrepancy between the data distribution used in the training
and testing of DR. In our experiments, ANCE boosts the BERT-Siamese DR
model to outperform all competitive dense and sparse retrieval baselines. It nearly
matches the accuracy of sparse-retrieval-and-BERT-reranking using dot-product in
the ANCE-learned representation space and provides almost 100x speed-up.

Our analyses further confirm that the negatives from sparse retrieval or other sampling methods differ
drastically from the actual negatives in DR, and that ANCE fundamentally resolves this mismatch.
We also show the influence of the asynchronous ANN refreshing on learning convergence and
demonstrate that the efficiency bottleneck is in the encoding update, not in the ANN part during
ANCE training. These qualifications demonstrate the advantages, perhaps also the necessity, of our
asynchronous ANCE learning in dense retrieval.

## Requirements

To install requirements, run the following commands:

```setup
git clone https://github.com/microsoft/ANCE
cd ANCE
python setup.py install
```

## Data Download
To download all the needed data, run:
```
bash commands/datadownload.sh 
```

## Data Preprocessing
The command to preprocess passage and document data is listed below:

```
python data/msmarco_data.py 
--data_dir $raw_data_dir \
--out_data_dir $preprocessed_data_dir \ 
--model_type {use rdot_nll for ANCE FirstP, rdot_nll_multi_chunk for ANCE MaxP} \ 
--model_name_or_path roberta-base \ 
--max_seq_length {use 512 for ANCE FirstP, 2048 for ANCE MaxP} \ 
--data_type {use 1 for passage, 0 for document}
```

The data preprocessing command is included as the first step in the training command file commands/run_train.sh

## Warmup for Training
ANCE training starts from a pretrained BM25 warmup checkpoint. The command with our used parameters to train this warmup checkpoint is in commands/run_train_warmup.py and is shown below:

        python3 -m torch.distributed.launch --nproc_per_node=1 ../drivers/run_warmup.py \
        --train_model_type rdot_nll \
        --model_name_or_path roberta-base \
        --task_name MSMarco \
        --do_train \
        --evaluate_during_training \
        --data_dir ${location of your raw data}  
        --max_seq_length 128 
        --per_gpu_eval_batch_size=256 \
        --per_gpu_train_batch_size=32 \
        --learning_rate 2e-4  \
        --logging_steps 100   \
        --num_train_epochs 2.0  \
        --output_dir ${location for checkpoint saving} \
        --warmup_steps 1000  \
        --overwrite_output_dir \
        --save_steps 30000 \
        --gradient_accumulation_steps 1 \
        --expected_train_size 35000000 \
        --logging_steps_per_eval 1 \
        --fp16 \
        --optimizer lamb \
        --log_dir ~/tensorboard/${DLWS_JOB_ID}/logs/OSpass

## Training

To train the model(s) in the paper, you need to start two commands in the following order:

1. run commands/run_train.sh which does three things in a sequence:

	a. Data preprocessing: this is explained in the previous data preprocessing section. This step will check if the preprocess data folder exists, and will be skipped if the checking is positive.

	b. Initial ANN data generation: this step will use the pretrained BM25 warmup checkpoint to generate the initial training data. The command is as follow:

        python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen.py 
        --training_dir {# checkpoint location, not used for initial data generation} \ 
        --init_model_dir {pretrained BM25 warmup checkpoint location} \ 
        --model_type rdot_nll \
        --output_dir $model_ann_data_dir \
        --cache_dir $model_ann_data_dir_cache \
        --data_dir $preprocessed_data_dir \
        --max_seq_length 512 \
        --per_gpu_eval_batch_size 16 \
        --topk_training {top k candidates for ANN search(ie:200)} \ 
        --negative_sample {negative samples per query(20)} \ 
        --end_output_num 0 # only set as 0 for initial data generation, do not set this otherwise

	c. Training: ANCE training with the most recently generated ANN data, the command is as follow:

        python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann.py 
        --model_type rdot_nll \
        --model_name_or_path $pretrained_checkpoint_dir \
        --task_name MSMarco \
        --triplet {# default = False, action="store_true", help="Whether to run training}\ 
        --data_dir $preprocessed_data_dir \
        --ann_dir {location of the ANN generated training data} \ 
        --max_seq_length 512 \
        --per_gpu_train_batch_size=8 \
        --gradient_accumulation_steps 2 \
        --learning_rate 1e-6 \
        --output_dir $model_dir \
        --warmup_steps 5000 \
        --logging_steps 100 \
        --save_steps 10000 \
        --optimizer lamb 
		
2. Once training starts, start another job in parallel to fetch the latest checkpoint from the ongoing training and update the training data. To do that, run

        bash commands/run_ann_data_gen.sh

    The command is similar to the initial ANN data generation command explained previously

## Inference
The command for inferencing query and passage/doc embeddings is the same as that for Initial ANN data generation described above as the first step in ANN data generation is inference. However you need to add --inference to the command to have the program to stop after the initial inference step. commands/run_inference.sh provides a sample command.

## Evaluation

The evaluation is done through "Calculate Metrics.ipynb". This notebook calculates full ranking and reranking metrics used in the paper including NDCG, MRR, hole rate, recall for passage/document, dev/eval set specified by user. In order to run it, you need to define the following parameters at the beginning of the Jupyter notebook.
        
        checkpoint_path = {location for dumpped query and passage/document embeddings which is output_dir from run_ann_data_gen.py}
        checkpoint =  {embedding from which checkpoint(ie: 200000)}
        data_type =  {0 for document, 1 for passage}
        test_set =  {0 for MSMARCO dev_set, 1 for TREC eval_set}
        raw_data_dir = 
        processed_data_dir = 

## ANCE VS DPR on OpenQA Benchmarks
We also evaluate ANCE on the OpenQA benchmark used in a parallel work ([DPR](https://github.com/facebookresearch/DPR)). At the time of our experiment, only the pre-processed NQ and TriviaQA data are released. 
Our experiments use the two released tasks and inherit DPR retriever evaluation. The evaluation uses the Coverage@20/100 which is whether the Top-20/100 retrieved passages include the answer. We explain the steps to 
reproduce our results on OpenQA Benchmarks in this section.

### Download data
commands/data_download.sh takes care of this step.

### ANN data generation & ANCE training
Following the same training philosophy discussed before, the ann data generation and ANCE training for OpenQA require two parallel jobs.
1. We need to preprocess data and generate an initial training set for ANCE to start training. The command for that is provided in:
```
commands/run_ann_data_gen_dpr.sh
```
We keep this data generation job running after it creates an initial training set as it will later keep generating training data with newest checkpoints from the training process.

2. After an initial training set is generated, we start an ANCE training job with commands provided in:
```
commands/run_train_dpr.sh
```
During training, the evaluation metrics will be printed to tensorboards each time it receives new training data. Alternatively, you could check the metrics in the dumped file "ann_ndcg_#" in the directory specified by "model_ann_data_dir" in commands/run_ann_data_gen_dpr.sh each time new training data is generated.

## Results
The run_train.sh and run_ann_data_gen.sh files contain the command with the parameters we used for passage ANCE(FirstP), document ANCE(FirstP) and document ANCE(MaxP)
Our model achieves the following performance on MSMARCO dev set and TREC eval set :


|   MSMARCO Dev Passage Retrieval    | MRR@10  | Recall@1k | Steps |
|---------------- | -------------- |-------------- | -------------- |
| ANCE(FirstP)   |     0.330         |      0.959       |      [600K](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Passage_ANCE_FirstP_Checkpoint.zip)       |
| ANCE(MaxP)   |     -         |      -       |      -       |

|   TREC DL Passage NDCG@10    | Rerank  | Retrieval | Steps |
|---------------- | -------------- |-------------- | -------------- |
| ANCE(FirstP)   |     0.677   |     0.648       |      [600K](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Passage_ANCE_FirstP_Checkpoint.zip)      |
| ANCE(MaxP)   |      -         |     -     |      -       |

|   TREC DL Document NDCG@10    | Rerank  | Retrieval | Steps |
|---------------- | -------------- |-------------- | -------------- |
| ANCE(FirstP)   |     0.641       |      0.615       |      [210K](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Document_ANCE_FirstP_Checkpoint.zip)       |
| ANCE(MaxP)   |      0.671       |      0.628    |      [139K](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Document_ANCE_MaxP_Checkpoint.zip)       |

|   MSMARCO Dev Passage Retrieval    | MRR@10  |  Steps |
|---------------- | -------------- | -------------- |
| pretrained BM25 warmup checkpoint   |     0.311       |       [60K](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/warmup_checpoint.zip)       |

| ANCE Single-task Training      | Top-20  |  Top-100 | Steps |
|---------------- | -------------- | -------------- |-------------- |
| NQ    |     81.9       |        87.5     |      [136K](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/nq.cp)       |
| TriviaQA    |    80.3         |       85.3       |      [100K](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/trivia.cp)       |

| ANCE Multi-task Training      | Top-20  |  Top-100 | Steps |
|---------------- | -------------- | -------------- |-------------- |
| NQ    |     82.1       |         87.9     |      [300K](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/multi.cp)       |
| TriviaQA    |    80.3        |       85.2       |      [300K](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/multi.cp)       |


Click the steps in the table to download the corresponding checkpoints.

Our result for document ANCE(FirstP) TREC eval set top 100 retrieved document per query could be downloaded [here](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Results/ance_512_eval_top100.txt).
Our result for document ANCE(MaxP) TREC eval set top 100 retrieved document per query could be downloaded [here](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Results/ance_2048_eval_top100.txt).

The TREC eval set query embedding and their ids for our passage ANCE(FirstP) experiment could be downloaded [here](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Passage_ANCE_FirstP_Embedding.zip).
The TREC eval set query embedding and their ids for our document ANCE(FirstP) experiment could be downloaded [here](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Document_ANCE_FirstP_Embedding.zip). 
The TREC eval set query embedding and their ids for our document 2048 ANCE(MaxP) experiment could be downloaded [here](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Document_ANCE_MaxP_Embedding.zip).

The t-SNE plots for all the queries in the TREC document eval set for ANCE(FirstP) could be viewed [here](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/t-SNE.zip).

run_train.sh and run_ann_data_gen.sh files contain the commands with the parameters we used for passage ANCE(FirstP), document ANCE(FirstP) and document 2048 ANCE(MaxP) to reproduce the results in this section.
run_train_warmup.sh contains the commands to reproduce the results for the pretrained BM25 warmup checkpoint in this section

Note the steps to reproduce similar results as shown in the table might be a little different due to different synchronizing between training and ann data generation processes and other possible environment differences of the user experiments.

∗Lee and Chenyan contributed equally
