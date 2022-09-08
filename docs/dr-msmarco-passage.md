# Training & Evaluating a Dense Retrieval Model on MS MARCO

## Data Download and Preprocess

Run the following command to download and extract the rocketqa-processed MS MARCO passage:

```bash
wget --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
tar -zxf marco.tar.gz
rm -rf marco.tar.gz
cd marco
```

In the same folder, download and extract the official MS MARCO files:

```bash
wget https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv -O qrels.train.tsv
gunzip qidpidtriples.train.full.2.tsv.gz
```

Merge the titles file and the paragraphs file together:

```bash
join  -t "$(echo -en '\t')"  -e '' -a 1  -o 1.1 2.2 1.2  <(sort -k1,1 para.txt) <(sort -k1,1 para.title.txt) | sort -k1,1 -n > corpus.tsv
```

Find out negatives from the triples file:

```bash
awk -v RS='\r\n' '$1==last {printf ",%s",$3; next} NR>1 {print "";} {last=$1; printf "%s\t%s",$1,$3;} END{print "";}' qidpidtriples.train.full.2.tsv > train.negatives.tsv
```

Now you can build the train files: 

```bash
python scripts/msmarco/build_train.py \
    --tokenizer_name $PLM_DIR/t5-base-scaled  \  # path to the HF tokenizer
    --negative_file $COLLECTION_DIR/marco/train.negatives.tsv  \  # the above negatives file
    --qrels $COLLECTION_DIR/marco/qrels.train.tsv  \
    --queries $COLLECTION_DIR/marco/train.query.txt  \
    --collection $COLLECTION_DIR/marco/corpus.tsv  \
    --save_to $PROCESSED_DIR/msmarco/t5/  \  # directory for output
    --doc_template "Title: <title> Text: <text>"  # passage-side template. <title> <text> will be replaced
```

The above step sample negatives from `train.negatives.tsv`, tokenize train queries and passages. It's recommended to merge all generated files into one file:

```bash
cat $PROCESSED_DIR/msmarco/t5/*.jsonl > $PROCESSED_DIR/msmarco/t5/train.jsonl
```

You may want to split a small portion from the train file for validation (optional):

```bash
(in the processed dir)
tail -n 500 train.jsonl > val.jsonl
head -n 400282 train.jsonl > train.new.jsonl
```

## Training and Evaluation

Start training:

```bash
python -m openmatch.driver.train_dr  \
    --output_dir $CHECKPOINT_DIR/msmarco/t5  \  # for checkpoints
    --model_name_or_path $PLM_DIR/t5-base-scaled  \  # HF PLM
    --do_train  \
    --save_steps 20000  \
    --eval_steps 20000  \
    --train_path $PROCESSED_DIR/msmarco/t5/train.new.jsonl  \
    --eval_path $PROCESSED_DIR/msmarco/t5/val.jsonl  \
    --fp16  \  # recommended
    --per_device_train_batch_size 8  \
    --train_n_passages 8  \
    --learning_rate 5e-6  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 3  \
    --logging_dir $LOG_DIR/msmarco/t5  \  # tensorboard logging dir
    --evaluation_strategy steps  # evaluate every `eval_steps` steps
```

The above command train a T5-based DR model for 3 epochs, evaluating it every 20,000 steps. Currently evaluation is just calculating the loss on the validation set; it may not correlate with actual retrieval performance very well. 

Note that the above sample uses T5 as the representation model. By default, the representation is extracted from the first token of the decoder output. To only use the encoder, set the `--encoder_only` flag on. You can also use other PLMs.

There are several useful arguments to adjust the model behavior:

|Name|Available Values|Description|
|-----|---|---|
|`--pooling`|`first` `mean`| Use the embedding of the first token as the representation, or use mean pooling of all tokens. (For encoder-only models)|
|`--add_linear_head`| | Set on to add a linear projection. You need to set `--projection_in_dim` and `projection_out_dim`.|
|`--normalize`| | Set on to normalize the embedding.|


After the first-round training, you can update the negatives using the trained model to get hard negatives. First you need to infer all the document embeddings based on the current model:

```bash
python -m openmatch.driver.build_index  \
    --output_dir $EMBEDDING_DIR/msmarco/t5/  \  # for embeddings
    --model_name_or_path $CHECKPOINT_DIR/msmarco/t5  \
    --per_device_eval_batch_size 256  \
    --corpus_path $COLLECTION_DIR/marco/corpus.tsv  \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len 128  \
    --fp16  \
    --dataloader_num_workers 1
```

The above step encode the whole `corpus.tsv` into document vectors. `corpus.tsv` is a tsv file, and we need to provide with column names using `--doc_column_names id,title,text`. The fields in the template will be replaced by values with corresponding names. In the above command, the first row of the tsv is recognized as the id, the second row as the title, the third row as the body text. The title and body text will be filled into the template to form the model input. Note that for training, this step is done during the preprocessing stage.

Retrieve document for train queries:

```bash
python -m openmatch.driver.retrieve  \
    --output_dir $EMBEDDING_DIR/msmarco/t5/  \
    --model_name_or_path $CHECKPOINT_DIR/msmarco/t5  \
    --per_device_eval_batch_size 256  \
    --query_path $COLLECTION_DIR/marco/train.query.txt  \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 32  \
    --fp16  \
    --trec_save_path $RESULT_DIR/marco/t5/train.trec  \  # TrecRun file for retrieval result, first create this directory
    --dataloader_num_workers 1
```

Then you can obtain hard negatives using the `build_hn.py` script:

```bash
python /path/to/openmatch/scripts/msmarco/build_hn.py  \
    --tokenizer_name $PLM_DIR/t5-base-scaled  \
    --hn_file $RESULT_DIR/marco/t5/train.trec  \
    --qrels $COLLECTION_DIR/marco/qrels.train.tsv  \
    --queries $COLLECTION_DIR/marco/train.query.txt  \
    --collection $COLLECTION_DIR/marco/corpus.tsv  \
    --save_to $PROCESSED_DIR/msmarco/t5/  \
    --doc_template "Title: <title> Text: <text>"
```

Merge all hard negatives files:

```bash
cat $PROCESSED_DIR/msmarco/t5/*.hn.jsonl > $PROCESSED_DIR/msmarco/t5/train.hn.jsonl
```

Split it:

```bash
(in the processed dir)
tail -n 500 train.hn.jsonl > val.hn.jsonl
head -n 502439 train.hn.jsonl > train.new.hn.jsonl
```

Train with hard negatives:

```bash
python -m openmatch.driver.train_dr  \
    --output_dir $CHECKPOINT_DIR/msmarco/t5_s2  \
    --model_name_or_path $CHECKPOINT_DIR/msmarco/t5  \  # start from stage 1 checkpoint
    --do_train  \
    --save_steps 20000  \
    --eval_steps 20000  \
    --train_path $PROCESSED_DIR/msmarco/t5/train.new.hn.jsonl  \
    --eval_path $PROCESSED_DIR/msmarco/t5/val.hn.jsonl  \
    --fp16  \
    --per_device_train_batch_size 8  \
    --train_n_passages 8  \
    --learning_rate 5e-6  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 3  \
    --logging_dir $LOG_DIR/msmarco/t5_s2  \
    --evaluation_strategy steps
```

Finally, perform retrieval on the official dev set:

```bash
python -m openmatch.driver.build_index  \
    --output_dir $EMBEDDING_DIR/msmarco/t5_s2/  \  # for embeddings
    --model_name_or_path $CHECKPOINT_DIR/msmarco/t5_s2  \
    --per_device_eval_batch_size 256  \
    --corpus_path $COLLECTION_DIR/marco/corpus.tsv  \
    --doc_template "Title: <title> Text: <text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len 128  \
    --fp16  \
    --dataloader_num_workers 1

python -m openmatch.driver.retrieve  \
    --output_dir $EMBEDDING_DIR/msmarco/t5_s2/  \
    --model_name_or_path $CHECKPOINT_DIR/msmarco/t5_s2  \
    --per_device_eval_batch_size 256  \
    --query_path $COLLECTION_DIR/marco/train.query.txt  \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 32  \
    --fp16  \
    --trec_save_path $RESULT_DIR/marco/t5-s2/dev.trec  \  # TrecRun file for retrieval result
    --dataloader_num_workers 1
```

## Notes

### T5 weights scaling

It's recommended to first scale the weights of T5 using `scripts/scale_t5_weights.py` before using it. The scaled version typically has a smoother optimization process. See `scale-t5-weights.md` for details.

### Distributed training and inference

Use `torch.distributed.launch` to start training and inference distributedly (i.e. using multiple GPUs), like:

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port 19286 -m openmatch.driver.train_dr/build_index ...
```

During distributed training, set `--negatives_x_device` flag on to share document embeddings through devices.

### Search with GPU(s)

If you've installed `faiss-gpu` properly, then you can perform search on GPU(s) by setting the `--use_gpu` flag on in `openmatch.driver.retrieve`.

### Issues with the dataloader

Currently setting `--dataloader_num_workers` greater than 1 will produce an error - the data will be duplicated. This is because we use `IterableDataset`, which has problems with vanilla multi-worker processing. We will fix it in the future.