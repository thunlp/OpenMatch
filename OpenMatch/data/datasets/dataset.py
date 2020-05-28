from typing import Union, List, Tuple, Dict, Any

import json

import torch
from torch.utils.data import Dataset

from OpenMatch.data.tokenizers import Tokenizer

class Dataset(Dataset):
    def __init__(
        self,
        dataset: Union[Dict, str],
        tokenizer: Tokenizer,
        mode: str,
        query_max_len: int = 10,
        doc_max_len: int = 256,
        max_input: int = 1280000,
        task: str = 'ranking'
    ) -> None:
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._mode = mode
        self._query_max_len = query_max_len
        self._doc_max_len = doc_max_len
        self._max_input = max_input
        self._task = task

        if isinstance(self._dataset, str):
            with open(self._dataset, 'r') as f:
                self._examples = []
                for i, line in enumerate(f):
                    if i >= self._max_input:
                        break
                    line = json.loads(line)
                    self._examples.append(line)
        elif isinstance(self._dataset, dict):
            queries = {}
            with open(self._dataset['queries'], 'r') as f:
                for line in f:
                    line = json.loads(line)
                    queries[line['query_id']] = line['query']
            docs = {}
            with open(self._dataset['docs'], 'r') as f:
                for line in f:
                    line = json.loads(line)
                    docs[line['doc_id']] = line['doc']
            if self._mode != 'test':
                qrels = {}
                with open(self._dataset['qrels'], 'r') as f:
                    for line in f:
                        line = line.strip().split()
                        if line[0] not in qrels:
                            qrels[line[0]] = {}
                        qrels[line[0]][line[2]] = int(line[3])
            with open(self._dataset['trec'], 'r') as f:
                self._examples = []
                for i, line in enumerate(f):
                    if i >= self._max_input:
                        break
                    line = line.strip().split()
                    if self._mode != 'test':
                        if line[0] not in qrels or line[2] not in qrels[line[0]]:
                            label = 0
                        else:
                            label = qrels[line[0]][line[2]]
                    if self._mode == 'train':
                        assert self._task == 'classification'
                        self._examples.append({'query': queries[line[0]], 'doc': docs[line[2]], 'label': label})
                    elif self._mode == 'dev':
                        self._examples.append({'query': queries[line[0]], 'doc': docs[line[2]], 'label': label, 'query_id': line[0], 'paper_id': line[2], 'retrieval_score': float(line[4])})
                    elif self._mode == 'test':
                        self._examples.append({'query': queries[line[0]], 'doc': docs[line[2]], 'query_id': line[0], 'paper_id': line[2], 'retrieval_score': float(line[4])})
                    else:
                        raise ValueError('Mode must be `train`, `dev` or `test`.')
        else:
            raise ValueError('Dataset must be `str` or `dict`.')
        self._count = len(self._examples)

    def collate(self, batch: Dict[str, Any]):
        if self._mode == 'train':
            if self._task == 'ranking':
                query_idx = torch.tensor([item['query_idx'] for item in batch])
                query_mask = torch.tensor([item['query_mask'] for item in batch])
                doc_pos_idx = torch.tensor([item['doc_pos_idx'] for item in batch])
                doc_pos_mask = torch.tensor([item['doc_pos_mask'] for item in batch])
                doc_neg_idx = torch.tensor([item['doc_neg_idx'] for item in batch])
                doc_neg_mask = torch.tensor([item['doc_neg_mask'] for item in batch])
                return {'query_idx': query_idx, 'query_mask': query_mask,
                        'doc_pos_idx': doc_pos_idx, 'doc_pos_mask': doc_pos_mask,
                        'doc_neg_idx': doc_neg_idx, 'doc_neg_mask': doc_neg_mask}
            elif self._task == 'classification':
                query_idx = torch.tensor([item['query_idx'] for item in batch])
                query_mask = torch.tensor([item['query_mask'] for item in batch])
                doc_idx = torch.tensor([item['doc_idx'] for item in batch])
                doc_mask = torch.tensor([item['doc_mask'] for item in batch])
                label = torch.tensor([item['label'] for item in batch])
                return {'query_idx': query_idx, 'query_mask': query_mask,
                        'doc_idx': doc_idx, 'doc_mask': doc_mask,
                        'label': label}
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
        elif self._mode == 'dev':
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            label = [item['label'] for item in batch]
            retrieval_score = torch.tensor([item['retrieval_score'] for item in batch])
            query_idx = torch.tensor([item['query_idx'] for item in batch]) 
            query_mask = torch.tensor([item['query_mask'] for item in batch])
            doc_idx = torch.tensor([item['doc_idx'] for item in batch])
            doc_mask = torch.tensor([item['doc_mask'] for item in batch])
            return {'query_id': query_id, 'doc_id': doc_id, 'label': label, 'retrieval_score': retrieval_score,
                    'query_idx': query_idx, 'query_mask': query_mask,
                    'doc_idx': doc_idx, 'doc_mask': doc_mask}
        else:
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            retrieval_score = torch.tensor([item['retrieval_score'] for item in batch])
            query_idx = torch.tensor([item['query_idx'] for item in batch])
            query_mask = torch.tensor([item['query_mask'] for item in batch])
            doc_idx = torch.tensor([item['doc_idx'] for item in batch])
            doc_mask = torch.tensor([item['doc_mask'] for item in batch])
            return {'query_id': query_id, 'doc_id': doc_id, 'retrieval_score': retrieval_score,
                    'query_idx': query_idx, 'query_mask': query_mask,
                    'doc_idx': doc_idx, 'doc_mask': doc_mask}

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        if self._mode == 'train':
            if self._task == 'ranking':
                query_idx, query_mask = self._tokenizer.process(example['query'], self._query_max_len)
                doc_pos_idx, doc_pos_mask = self._tokenizer.process(example['doc_pos'], self._doc_max_len)
                doc_neg_idx, doc_neg_mask = self._tokenizer.process(example['doc_neg'], self._doc_max_len)
                return {'query_idx': query_idx, 'query_mask': query_mask,
                        'doc_pos_idx': doc_pos_idx, 'doc_pos_mask': doc_pos_mask,
                        'doc_neg_idx': doc_neg_idx, 'doc_neg_mask': doc_neg_mask}
            elif self._task == 'classification':
                query_idx, query_mask = self._tokenizer.process(example['query'], self._query_max_len)
                doc_idx, doc_mask = self._tokenizer.process(example['doc'], self._doc_max_len)
                return {'query_idx': query_idx, 'query_mask': query_mask,
                        'doc_idx': doc_idx, 'doc_mask': doc_mask,
                        'label': example['label']}
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
        elif self._mode == 'dev':
            query_idx, query_mask = self._tokenizer.process(example['query'], self._query_max_len)
            doc_idx, doc_mask = self._tokenizer.process(example['doc'], self._doc_max_len)
            return {'query_id': example['query_id'], 'doc_id': example['paper_id'], 'label': example['label'], 'retrieval_score': example['retrieval_score'],
                    'query_idx': query_idx, 'query_mask': query_mask,
                    'doc_idx': doc_idx, 'doc_mask': doc_mask}
        elif self._mode == 'test':
            query_idx, query_mask = self._tokenizer.process(example['query'], self._query_max_len)
            doc_idx, doc_mask = self._tokenizer.process(example['doc'], self._doc_max_len)
            return {'query_id': example['query_id'], 'doc_id': example['paper_id'], 'retrieval_score': example['retrieval_score'],
                    'query_idx': query_idx, 'query_mask': query_mask,
                    'doc_idx': doc_idx, 'doc_mask': doc_mask}
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def __len__(self) -> int:
        return self._count
