from typing import List, Tuple, Dict, Any

import json

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

class BertMLMDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        tokenizer: AutoTokenizer,
        seq_max_len: int = 256,
        max_input: int = 1280000,
    ) -> None:
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._seq_max_len = seq_max_len
        self._max_input = max_input
        if self._seq_max_len > 512:
            raise ValueError('query_max_len + doc_max_len + 3 > 512.')

        with open(self._dataset, 'r') as f:
            self._examples = []
            for i, line in enumerate(f):
                if i >= self._max_input:
                    break
                line = json.loads(line)
                self._examples.append(line)
        self._count = len(self._examples)

    def collate(self, batch: Dict[str, Any]):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        segment_ids = torch.tensor([item['segment_ids'] for item in batch])
        input_mask = torch.tensor([item['input_mask'] for item in batch])
        return {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask}

    def pack_bert_features(self, doc_tokens: List[str]):
        input_tokens = [self._tokenizer.cls_token] + doc_tokens + [self._tokenizer.sep_token]
        input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_tokens)

        padding_len = self._seq_max_len - len(input_ids)
        input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
        input_mask = input_mask + [0] * padding_len
        segment_ids = segment_ids + [0] * padding_len

        assert len(input_ids) == self._seq_max_len
        assert len(input_mask) == self._seq_max_len
        assert len(segment_ids) == self._seq_max_len

        return input_ids, input_mask, segment_ids

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        doc_tokens = self._tokenizer.tokenize(example['doc'])[:self._seq_max_len - 2]

        input_ids, input_mask, segment_ids = self.pack_bert_features(doc_tokens)
        return {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids}

    def __len__(self) -> int:
        return self._count
