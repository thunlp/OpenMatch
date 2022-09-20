# Training & Evaluating a Re-ranking Model on MS MARCO

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
# BERT-like models
python scripts/msmarco/build_train.py \
    --tokenizer_name $PLM_DIR/t5-base-scaled  \  # path to the HF tokenizer
    --negative_file $COLLECTION_DIR/marco/train.negatives.tsv  \  # the above negatives file
    --qrels $COLLECTION_DIR/marco/qrels.train.tsv  \
    --queries $COLLECTION_DIR/marco/train.query.txt  \
    --collection $COLLECTION_DIR/marco/corpus.tsv  \
    --save_to $PROCESSED_DIR/msmarco/bert/  \  # directory for output
    --query_template "<text> [SEP]" \
    --doc_template "<text>"  # passage-side template. <title> <text> will be replaced
```

```bash
# monoT5
python scripts/msmarco/build_train.py \
    --tokenizer_name $PLM_DIR/t5-base-scaled  \  # path to the HF tokenizer
    --negative_file $COLLECTION_DIR/marco/train.negatives.tsv  \  # the above negatives file
    --qrels $COLLECTION_DIR/marco/qrels.train.tsv  \
    --queries $COLLECTION_DIR/marco/train.query.txt  \
    --collection $COLLECTION_DIR/marco/corpus.tsv  \
    --save_to $PROCESSED_DIR/msmarco/t5/  \  # directory for output
    --query_template "Query: <text>" \
    --doc_template "Document: <text> Relevant: "  # passage-side template. <title> <text> will be replaced
```

The query and document will be filled in the blanks, and be further concatenated during training.

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
python -m openmatch.driver.train_rr  \
    --output_dir $CHECKPOINT_DIR/msmarco/bert_bce  \
    --model_name_or_path $PLM_DIR/bert-base-uncased  \
    --do_train  \
    --save_steps 10000  \
    --train_path $PROCESSED_DIR/msmarco/bert/train.jsonl  \
    --eval_path $PROCESSED_DIR/msmarco/bert/eval.jsonl  \
    --eval_steps 10000  \
    --fp16  \
    --per_device_train_batch_size 32  \
    --per_device_eval_batch_size 128  \
    --learning_rate 1e-5  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 30  \
    --loss_fn bce  \
    --logging_dir $LOG_DIR/bert_bce  \
    --evaluation_strategy steps  \
    --dataloader_num_workers 1
```

The above command train a BERT-based re-ranking model for 30 epochs, evaluating it every 100,00 steps. Currently evaluation is just calculating the loss on the validation set; it may not correlate with actual re-ranking performance very well. 

You can set the loss function used during training via the `--loss_fn` argument, which can be set to one of the following values:

|Value|Description|Calculation (for a pair of pos & neg documents)|
|---|---|---|
|`mr`|Margin Ranking Loss (pairwise)|$\max (0, m - s_+ + s_-)$|
|`smr`|Soft Margin Ranking Loss (pairwise)|$\log (1 + \exp(m - s_+ + s_-))$|
|`bce`|Binary Cross-Entropy Loss (pointwise)|$-\log(\text{sigmoid}(s_+))-\log(1-\text{sigmoid}(s_-))$|

where $s_+$, $s_-$ are the scores of positive (q, d) pair and negative (q, d) pair, respectively, and $m$ is the margin controlled by the `--margin` argument.

The following command shows the training of monoT5:

```bash
python -m openmatch.driver.train_rr  \
    --output_dir $CHECKPOINT_DIR/msmarco/t5  \
    --model_name_or_path $PLM_DIR/t5-base-scaled  \
    --do_train  \
    --save_steps 10000  \
    --train_path $PROCESSED_DIR/msmarco/t5/train.jsonl  \
    --eval_path $PROCESSED_DIR/msmarco/t5/eval.jsonl  \
    --eval_steps 10000  \
    --fp16  \
    --per_device_train_batch_size 32  \
    --per_device_eval_batch_size 128  \
    --learning_rate 1e-5  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 30  \
    --pos_token true  \
    --neg_token false  \
    --logging_dir $LOG_DIR/t5  \
    --evaluation_strategy steps  \
    --dataloader_num_workers 1
```

Following Nogueira et al., we use the template "Query: [Q] Document: [D] Relevant:" during preprocessing, and the model is fine-tuned to produce the tokens "true" or "false". Thus we set `--pos_token` to `true` and `--neg_token` to `false`. The loss of monoT5 is set to cross entropy over the softmax on the logits of the two tokens and cannot be changed.

After training, test your model using the following command:

```bash
python -m openmatch.driver.rerank  \
    --output_dir None  \
    --model_name_or_path $CHECKPOINT_DIR/msmarco/bert_bce  \
    --per_device_eval_batch_size 256  \
    --query_path $COLLECTION_DIR/marco/dev.query.txt  \
    --corpus_path .$COLLECTION_DIR/marco/corpus.tsv  \
    --query_template "<text> [SEP]"  \
    --doc_template "<text>"  \
    --query_column_names id,text  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len 128  \
    --fp16  \
    --trec_run_path /path/to/first-stage/trecrun/file  \
    --trec_save_path $RESULT_DIR/rr.trec  \
    --dataloader_num_workers 1 
```

where `--trec_run_path` is the path to a run file in TREC format, produced by a previous-stage ranker (e.g. BM25). The command for monoT5 is as similar as just replacing `--model_name_or_path`.