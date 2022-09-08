# Adapted from Tevatron (https://github.com/texttron/tevatron)

import json
import os

from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer

from ..arguments import DataArguments
from ..utils import fill_template, find_all_markers


def get_idx(obj):
    example_id = obj.get("_id", None) or obj.get("id", None)
    example_id = str(example_id) if example_id is not None else None
    return example_id


class InferenceDataset(IterableDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, is_query: bool = False, final: bool = True, cache_dir: str = None):
        super(InferenceDataset, self).__init__()
        self.cache_dir = cache_dir
        self.processed_data_path = data_args.processed_data_path
        self.data_files = [data_args.query_path] if is_query else [data_args.corpus_path]
        self.tokenizer = tokenizer
        self.max_len = data_args.q_max_len if is_query else data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.template = data_args.query_template if is_query else data_args.doc_template
        self.all_markers = find_all_markers(self.template)
        self.stream = not data_args.map_style
        self.final = final

    @classmethod
    def load(cls, tokenizer: PreTrainedTokenizer, data_args: DataArguments, is_query: bool = False, final: bool = True, cache_dir: str = None):
        data_files = [data_args.query_path] if is_query else [data_args.corpus_path]
        ext = os.path.splitext(data_files[0])[1]
        if ext == ".jsonl":
            return JsonlDataset(tokenizer, data_args, is_query, final, cache_dir)
        elif ext in [".tsv", ".txt"]:
            return TsvDataset(tokenizer, data_args, is_query, final, cache_dir)
        else:
            raise ValueError("Unsupported dataset file extension {}".format(ext))

    def _process_func(self, example):
        example_id = get_idx(example)
        full_text = fill_template(self.template, example, self.all_markers, allow_not_found=True)
        tokenized = self.tokenizer(full_text, add_special_tokens=self.final, padding='max_length' if self.final else False, truncation=True, max_length=self.max_len, return_attention_mask=self.final, return_token_type_ids=self.final)
        return {"text_id": example_id, **tokenized}

    def __iter__(self):
        return iter(self.dataset.map(self._process_func, remove_columns=self.all_columns))

    def __getitem__(self, index):
        return self._process_func(self.dataset[index])


class JsonlDataset(InferenceDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, is_query: bool = False, final: bool = True, cache_dir: str = None):
        super(JsonlDataset, self).__init__(tokenizer, data_args, is_query, final, cache_dir)
        if self.stream:
            self.dataset = load_dataset("json", data_files=self.data_files, streaming=self.stream, cache_dir=cache_dir)["train"]
            sample = list(self.dataset.take(1))[0]
            self.all_columns = sample.keys()
        else:
            self.dataset = {}
            with open(self.data_files[0], "r") as f:
                for line in f:
                    obj = json.loads(line)
                    example_id = get_idx(obj)
                    self.dataset[example_id] = obj
                    self.all_columns = obj.keys()


class TsvDataset(InferenceDataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, is_query: bool = False, final: bool = True, cache_dir: str = None):
        super(TsvDataset, self).__init__(tokenizer, data_args, is_query, final, cache_dir)
        self.all_columns = data_args.query_column_names if is_query else data_args.doc_column_names
        self.all_columns = self.all_columns.split(',')
        if self.stream:
            self.dataset = load_dataset(
                "csv", 
                data_files=self.data_files, 
                streaming=self.stream, 
                column_names=self.all_columns,
                delimiter='\t',
                cache_dir=cache_dir
            )["train"]
        else:
            self.dataset = {}
            with open(self.data_files[0], "r") as f:
                for line in f:
                    all_contents = line.strip().split("\t")
                    obj = {}
                    for key, value in zip(self.all_columns, all_contents):
                        obj[key] = value
                    example_id = get_idx(obj)
                    self.dataset[example_id] = obj
        
