from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import BatchEncoding, PreTrainedTokenizer
import os
from transformers.data.data_collator import DefaultDataCollator
from dataclasses import dataclass

from ..arguments import DataArguments

class BEIRDataset:

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str = None):
        self.corpus_dataset = BEIRCorpusDataset(tokenizer, data_args, os.path.join(data_args.data_dir, "corpus.jsonl"), cache_dir)
        qrel_path = os.path.join(data_args.data_dir, "qrels", "test.tsv")
        self.qrel = {}
        with open(qrel_path, "r") as f:
            next(iter(f))  # omit title line
            for line in f:
                qid, docid, rel = line.strip().split()
                rel = int(rel)
                if qid not in self.qrel:
                    self.qrel[qid] = {docid: rel}
                else:
                    self.qrel[qid][docid] = rel
        self.query_dataset = BEIRQueryDataset(tokenizer, data_args, os.path.join(data_args.data_dir, "queries.jsonl"), list(self.qrel.keys()), cache_dir)


class BEIRQueryDataset(IterableDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, file_path: str, qids: list, cache_dir: str = None):
        super(BEIRQueryDataset, self).__init__()
        self.dataset = load_dataset("json", data_files=file_path, streaming=True, cache_dir=cache_dir)["train"]
        self.dataset = self.dataset.filter(lambda example: example["_id"] in qids)
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.proc_num = data_args.dataset_proc_num
        self.iter = None

    def __iter__(self):
        # print("iter")
        def process_func(examples):
            example_ids = examples["_id"]
            texts = examples["text"]
            tokenized = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.q_max_len)
            return BatchEncoding({"text_id": example_ids, **tokenized})

        self.iter = iter(self.dataset.map(
                        process_func, 
                        batched=True, 
                        remove_columns=["_id", "text", "metadata"], 
                        # desc="Running tokenizer on dataset",
                    ))

        return self.iter


class BEIRCorpusDataset(IterableDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, file_path: str, cache_dir: str):
        super(BEIRCorpusDataset, self).__init__()
        self.dataset = load_dataset("json", data_files=file_path, streaming=True, cache_dir=cache_dir)["train"]
        self.tokenizer = tokenizer
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.template = getattr(self.tokenizer, data_args.template, data_args.template)
        self.iter = None

        sample = list(self.dataset.take(1))[0]
        self.all_columns = sample.keys()

    def __iter__(self):
        def process_func(examples):
            example_ids = examples["_id"]
            texts = examples["text"]
            titles = examples.get("title", None)
            if titles is not None:
                texts_with_title = []
                for text, title in zip(texts, titles):
                    if title.strip() == "":
                        title = "-"
                    text_with_title = self.template.replace("<title>", title).replace("<text>", text)
                    texts_with_title.append(text_with_title)
                tokenized = self.tokenizer(texts_with_title, padding='max_length', truncation=True, max_length=self.p_max_len)
            else:
                texts_with_title = []
                for text, title in zip(texts, titles):
                    text_with_title = self.template.replace("<title>", "-").replace("<text>", text)
                    texts_with_title.append(text_with_title)
                tokenized = self.tokenizer(texts_with_title, padding='max_length', truncation=True, max_length=self.p_max_len)
            return BatchEncoding({"text_id": example_ids, **tokenized})

        self.iter = iter(self.dataset.map(
                        process_func, 
                        batched=True, 
                        remove_columns=self.all_columns, 
                        # desc="Running tokenizer on dataset",
                    ))
        return self.iter