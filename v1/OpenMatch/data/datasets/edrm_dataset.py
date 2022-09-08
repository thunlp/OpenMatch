from typing import Union, List, Tuple, Dict, Any

import json

import torch
from torch.utils.data import Dataset

from OpenMatch.data.tokenizers import Tokenizer

class EDRMDataset(Dataset):
    def __init__(
        self,
        dataset: Union[Dict, str],
        wrd_tokenizer: Tokenizer,
        ent_tokenizer: Tokenizer,
        mode: str,
        query_max_len: int = 10,
        doc_max_len: int = 256,
        des_max_len: int = 20,
        max_ent_num: int = 3,
        max_input: int = 1280000,
        task: str = 'ranking'
    ) -> None:
        self._dataset = dataset
        self._wrd_tokenizer = wrd_tokenizer
        self._ent_tokenizer = ent_tokenizer
        self._mode = mode
        self._query_max_len = query_max_len
        self._doc_max_len = doc_max_len
        self._des_max_len = des_max_len
        self._max_ent_num = max_ent_num
        self._max_input = max_input
        self._task = task

        if isinstance(self._dataset, str):
            self._id = False
            with open(self._dataset, 'r') as f:
                self._examples = []
                for i, line in enumerate(f):
                    if i >= self._max_input:
                        break
                    if self._mode != 'train' or self._dataset.split('.')[-1] == 'json' or self._dataset.split('.')[-1] == 'jsonl':
                        line = json.loads(line)
                    else:
                        if self._task == 'ranking':
                            query, doc_pos, doc_neg = line.strip('\n').split('\t')
                            line = {'query': query, 'doc_pos': doc_pos, 'doc_neg': doc_neg}
                        elif self._task == 'classification':
                            query, doc, label = line.strip('\n').split('\t')
                            line = {'query': query, 'doc': doc, 'label': int(label)}
                        else:
                            raise ValueError('Task must be `ranking` or `classification`.')
                    self._examples.append(line)
        elif isinstance(self._dataset, dict):
            self._id = True
            self._queries = {}
            with open(self._dataset['queries'], 'r') as f:
                for line in f:
                    if self._dataset['queries'].split('.')[-1] == 'json' or self._dataset['queries'].split('.')[-1] == 'jsonl':
                        line = json.loads(line)
                    else:
                        query_id, query = line.strip('\n').split('\t')
                        line = {'query_id': query_id, 'query': query}
                    self._queries[line['query_id']] = (line['query'], line['query_ent'], line['query_des'])
            self._docs = {}
            with open(self._dataset['docs'], 'r') as f:
                for line in f:
                    if self._dataset['docs'].split('.')[-1] == 'json' or self._dataset['docs'].split('.')[-1] == 'jsonl':
                        line = json.loads(line)
                    else:
                        doc_id, doc = line.strip('\n').split('\t')
                        line = {'doc_id': doc_id, 'doc': doc}
                    self._docs[line['doc_id']] = (line['doc'], line['doc_ent'], line['doc_des'])
            if self._mode == 'dev':
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
                    if self._mode == 'dev':
                        if line[0] not in qrels or line[2] not in qrels[line[0]]:
                            label = 0
                        else:
                            label = qrels[line[0]][line[2]]
                    if self._mode == 'train':
                        if self._task == 'ranking':
                            self._examples.append({'query_id': line[0], 'doc_pos_id': line[1], 'doc_neg_id': line[2]})
                        elif self._task == 'classification':
                            self._examples.append({'query': line[0], 'doc_id': line[2], 'label': int(line[2])})
                        else:
                            raise ValueError('Task must be `ranking` or `classification`.')
                    elif self._mode == 'dev':
                        self._examples.append({'label': label, 'query_id': line[0], 'doc_id': line[2], 'retrieval_score': float(line[4])})
                    elif self._mode == 'test':
                        self._examples.append({'query_id': line[0], 'doc_id': line[2], 'retrieval_score': float(line[4])})
                    else:
                        raise ValueError('Mode must be `train`, `dev` or `test`.')
        else:
            raise ValueError('Dataset must be `str` or `dict`.')
        self._count = len(self._examples)

    def collate(self, batch: Dict[str, Any]):
        if self._mode == 'train':
            if self._task == 'ranking':
                query_wrd_idx = torch.tensor([item['query_wrd_idx'] for item in batch])
                query_wrd_mask = torch.tensor([item['query_wrd_mask'] for item in batch])
                doc_pos_wrd_idx = torch.tensor([item['doc_pos_wrd_idx'] for item in batch])
                doc_pos_wrd_mask = torch.tensor([item['doc_pos_wrd_mask'] for item in batch])
                doc_neg_wrd_idx = torch.tensor([item['doc_neg_wrd_idx'] for item in batch])
                doc_neg_wrd_mask = torch.tensor([item['doc_neg_wrd_mask'] for item in batch])
                query_ent_idx = torch.tensor([item['query_ent_idx'] for item in batch])
                query_ent_mask = torch.tensor([item['query_ent_mask'] for item in batch])
                doc_pos_ent_idx = torch.tensor([item['doc_pos_ent_idx'] for item in batch])
                doc_pos_ent_mask = torch.tensor([item['doc_pos_ent_mask'] for item in batch])
                doc_neg_ent_idx = torch.tensor([item['doc_neg_ent_idx'] for item in batch])
                doc_neg_ent_mask = torch.tensor([item['doc_neg_ent_mask'] for item in batch])
                query_des_idx = torch.tensor([item['query_des_idx'] for item in batch])
                doc_pos_des_idx = torch.tensor([item['doc_pos_des_idx'] for item in batch])
                doc_neg_des_idx = torch.tensor([item['doc_neg_des_idx'] for item in batch])
                return {'query_wrd_idx': query_wrd_idx, 'query_wrd_mask': query_wrd_mask,
                        'doc_pos_wrd_idx': doc_pos_wrd_idx, 'doc_pos_wrd_mask': doc_pos_wrd_mask,
                        'doc_neg_wrd_idx': doc_neg_wrd_idx, 'doc_neg_wrd_mask': doc_neg_wrd_mask,
                        'query_ent_idx': query_ent_idx, 'query_ent_mask': query_ent_mask,
                        'doc_pos_ent_idx': doc_pos_ent_idx, 'doc_pos_ent_mask': doc_pos_ent_mask,
                        'doc_neg_ent_idx': doc_neg_ent_idx, 'doc_neg_ent_mask': doc_neg_ent_mask,
                        'query_des_idx': query_des_idx, 'doc_pos_des_idx': doc_pos_des_idx, 'doc_neg_des_idx': doc_neg_des_idx}
            elif self._task == 'classification':
                query_wrd_idx = torch.tensor([item['query_wrd_idx'] for item in batch])
                query_wrd_mask = torch.tensor([item['query_wrd_mask'] for item in batch])
                doc_wrd_idx = torch.tensor([item['doc_wrd_idx'] for item in batch])
                doc_wrd_mask = torch.tensor([item['doc_wrd_mask'] for item in batch])
                query_ent_idx = torch.tensor([item['query_ent_idx'] for item in batch])
                query_ent_mask = torch.tensor([item['query_ent_mask'] for item in batch])
                doc_ent_idx = torch.tensor([item['doc_ent_idx'] for item in batch])
                doc_ent_mask = torch.tensor([item['doc_ent_mask'] for item in batch])
                query_des_idx = torch.tensor([item['query_des_idx'] for item in batch])
                doc_des_idx = torch.tensor([item['doc_des_idx'] for item in batch])
                label = torch.tensor([item['label'] for item in batch])
                return {'query_wrd_idx': query_wrd_idx, 'query_wrd_mask': query_wrd_mask,
                        'doc_wrd_idx': doc_wrd_idx, 'doc_wrd_mask': doc_wrd_mask,
                        'query_ent_idx': query_ent_idx, 'query_ent_mask': query_ent_mask,
                        'doc_ent_idx': doc_ent_idx, 'doc_ent_mask': doc_ent_mask,
                        'query_des_idx': query_des_idx, 'doc_des_idx': doc_des_idx,
                        'label': label}
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
        elif self._mode == 'dev':
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            label = [item['label'] for item in batch]
            retrieval_score = torch.tensor([item['retrieval_score'] for item in batch])
            query_wrd_idx = torch.tensor([item['query_wrd_idx'] for item in batch])
            query_wrd_mask = torch.tensor([item['query_wrd_mask'] for item in batch])
            doc_wrd_idx = torch.tensor([item['doc_wrd_idx'] for item in batch])
            doc_wrd_mask = torch.tensor([item['doc_wrd_mask'] for item in batch])
            query_ent_idx = torch.tensor([item['query_ent_idx'] for item in batch])
            query_ent_mask = torch.tensor([item['query_ent_mask'] for item in batch])
            doc_ent_idx = torch.tensor([item['doc_ent_idx'] for item in batch])
            doc_ent_mask = torch.tensor([item['doc_ent_mask'] for item in batch])
            query_des_idx = torch.tensor([item['query_des_idx'] for item in batch])
            doc_des_idx = torch.tensor([item['doc_des_idx'] for item in batch])
            return {'query_id': query_id, 'doc_id': doc_id, 'label': label, 'retrieval_score': retrieval_score,
                    'query_wrd_idx': query_wrd_idx, 'query_wrd_mask': query_wrd_mask,
                    'doc_wrd_idx': doc_wrd_idx, 'doc_wrd_mask': doc_wrd_mask,
                    'query_ent_idx': query_ent_idx, 'query_ent_mask': query_ent_mask,
                    'doc_ent_idx': doc_ent_idx, 'doc_ent_mask': doc_ent_mask,
                    'query_des_idx': query_des_idx, 'doc_des_idx': doc_des_idx}
        else:
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            retrieval_score = torch.tensor([item['retrieval_score'] for item in batch])
            query_wrd_idx = torch.tensor([item['query_wrd_idx'] for item in batch])
            query_wrd_mask = torch.tensor([item['query_wrd_mask'] for item in batch])
            doc_wrd_idx = torch.tensor([item['doc_wrd_idx'] for item in batch])
            doc_wrd_mask = torch.tensor([item['doc_wrd_mask'] for item in batch])
            query_ent_idx = torch.tensor([item['query_ent_idx'] for item in batch])
            query_ent_mask = torch.tensor([item['query_ent_mask'] for item in batch])
            doc_ent_idx = torch.tensor([item['doc_ent_idx'] for item in batch])
            doc_ent_mask = torch.tensor([item['doc_ent_mask'] for item in batch])
            query_des_idx = torch.tensor([item['query_des_idx'] for item in batch])
            doc_des_idx = torch.tensor([item['doc_des_idx'] for item in batch])
            return {'query_id': query_id, 'doc_id': doc_id, 'retrieval_score': retrieval_score,
                    'query_wrd_idx': query_wrd_idx, 'query_wrd_mask': query_wrd_mask,
                    'doc_wrd_idx': doc_wrd_idx, 'doc_wrd_mask': doc_wrd_mask,
                    'query_ent_idx': query_ent_idx, 'query_ent_mask': query_ent_mask,
                    'doc_ent_idx': doc_ent_idx, 'doc_ent_mask': doc_ent_mask,
                    'query_des_idx': query_des_idx, 'doc_des_idx': doc_des_idx}

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        if self._id:
            example['query'], example['query_ent'], example['query_des'] = self._queries[example['query_id']]
            if self._mode == 'train' and self._task == 'ranking':
                example['doc_pos'], example['doc_pos_ent'], example['doc_pos_des'] = self._docs[example['doc_pos_id']]
                example['doc_neg'], example['doc_neg_ent'], example['doc_neg_des'] = self._docs[example['doc_neg_id']]
            else:
                example['doc'], example['doc_ent'], example['doc_des'] = self._docs[example['doc_id']]
        if self._mode == 'train':
            if self._task == 'ranking':
                query_wrd_idx, query_wrd_mask = self._wrd_tokenizer.process(example['query'], self._query_max_len)
                doc_pos_wrd_idx, doc_pos_wrd_mask = self._wrd_tokenizer.process(example['doc_pos'], self._doc_max_len)
                doc_neg_wrd_idx, doc_neg_wrd_mask = self._wrd_tokenizer.process(example['doc_neg'], self._doc_max_len)
                query_ent_idx, query_ent_mask = self._ent_tokenizer.token_process(example['query_ent'], self._max_ent_num)
                doc_pos_ent_idx, doc_pos_ent_mask = self._ent_tokenizer.token_process(example['doc_pos_ent'], self._max_ent_num)
                doc_neg_ent_idx, doc_neg_ent_mask = self._ent_tokenizer.token_process(example['doc_neg_ent'], self._max_ent_num)
                query_des_idx, query_des_mask = self._wrd_tokenizer.batch_process(example['query_des'], self._des_max_len, self._max_ent_num)
                doc_pos_des_idx, doc_pos_des_mask = self._wrd_tokenizer.batch_process(example['doc_pos_des'], self._des_max_len, self._max_ent_num)
                doc_neg_des_idx, doc_neg_des_mask = self._wrd_tokenizer.batch_process(example['doc_neg_des'], self._des_max_len, self._max_ent_num)
                return {'query_wrd_idx': query_wrd_idx, 'query_wrd_mask': query_wrd_mask,
                        'doc_pos_wrd_idx': doc_pos_wrd_idx, 'doc_pos_wrd_mask': doc_pos_wrd_mask,
                        'doc_neg_wrd_idx': doc_neg_wrd_idx, 'doc_neg_wrd_mask': doc_neg_wrd_mask,
                        'query_ent_idx': query_ent_idx, 'query_ent_mask': query_ent_mask,
                        'doc_pos_ent_idx': doc_pos_ent_idx, 'doc_pos_ent_mask': doc_pos_ent_mask,
                        'doc_neg_ent_idx': doc_neg_ent_idx, 'doc_neg_ent_mask': doc_neg_ent_mask,
                        'query_des_idx': query_des_idx, 'doc_pos_des_idx': doc_pos_des_idx, 'doc_neg_des_idx': doc_neg_des_idx}
            elif self._task == 'classification':
                query_wrd_idx, query_wrd_mask = self._wrd_tokenizer.process(example['query'], self._query_max_len)
                doc_wrd_idx, doc_wrd_mask = self._wrd_tokenizer.process(example['doc'], self._doc_max_len)
                query_ent_idx, query_ent_mask = self._ent_tokenizer.token_process(example['query_ent'], self._max_ent_num)
                doc_ent_idx, doc_ent_mask = self._ent_tokenizer.token_process(example['doc_ent'], self._max_ent_num)
                query_des_idx, query_des_mask = self._wrd_tokenizer.batch_process(example['query_des'], self._des_max_len, self._max_ent_num)
                doc_des_idx, doc_des_mask = self._wrd_tokenizer.batch_process(example['doc_des'], self._des_max_len, self._max_ent_num)
                return {'query_wrd_idx': query_wrd_idx, 'query_wrd_mask': query_wrd_mask,
                        'doc_wrd_idx': doc_wrd_idx, 'doc_wrd_mask': doc_wrd_mask,
                        'query_ent_idx': query_ent_idx, 'query_ent_mask': query_ent_mask,
                        'doc_ent_idx': doc_ent_idx, 'doc_ent_mask': doc_ent_mask,
                        'query_des_idx': query_des_idx, 'doc_des_idx': doc_des_idx,
                        'label': example['label']}
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
        elif self._mode == 'dev':
            query_wrd_idx, query_wrd_mask = self._wrd_tokenizer.process(example['query'], self._query_max_len)
            doc_wrd_idx, doc_wrd_mask = self._wrd_tokenizer.process(example['doc'], self._doc_max_len)
            query_ent_idx, query_ent_mask = self._ent_tokenizer.token_process(example['query_ent'], self._max_ent_num)
            doc_ent_idx, doc_ent_mask = self._ent_tokenizer.token_process(example['doc_ent'], self._max_ent_num)
            query_des_idx, query_des_mask = self._wrd_tokenizer.batch_process(example['query_des'], self._des_max_len, self._max_ent_num)
            doc_des_idx, doc_des_mask = self._wrd_tokenizer.batch_process(example['doc_des'], self._des_max_len, self._max_ent_num)
            return {'query_id': example['query_id'], 'doc_id': example['doc_id'], 'label': example['label'], 'retrieval_score': example['retrieval_score'],
                    'query_wrd_idx': query_wrd_idx, 'query_wrd_mask': query_wrd_mask,
                    'doc_wrd_idx': doc_wrd_idx, 'doc_wrd_mask': doc_wrd_mask,
                    'query_ent_idx': query_ent_idx, 'query_ent_mask': query_ent_mask,
                    'doc_ent_idx': doc_ent_idx, 'doc_ent_mask': doc_ent_mask,
                    'query_des_idx': query_des_idx, 'doc_des_idx': doc_des_idx}
        elif self._mode == 'test':
            query_wrd_idx, query_wrd_mask = self._wrd_tokenizer.process(example['query'], self._query_max_len)
            doc_wrd_idx, doc_wrd_mask = self._wrd_tokenizer.process(example['doc'], self._doc_max_len)
            query_ent_idx, query_ent_mask = self._ent_tokenizer.token_process(example['query_ent'], self._max_ent_num)
            doc_ent_idx, doc_ent_mask = self._ent_tokenizer.token_process(example['doc_ent'], self._max_ent_num)
            query_des_idx, query_des_mask = self._wrd_tokenizer.batch_process(example['query_des'], self._des_max_len, self._max_ent_num)
            doc_des_idx, doc_des_mask = self._wrd_tokenizer.batch_process(example['doc_des'], self._des_max_len, self._max_ent_num)
            return {'query_id': example['query_id'], 'doc_id': example['doc_id'], 'retrieval_score': example['retrieval_score'],
                    'query_wrd_idx': query_wrd_idx, 'query_wrd_mask': query_wrd_mask,
                    'doc_wrd_idx': doc_wrd_idx, 'doc_wrd_mask': doc_wrd_mask,
                    'query_ent_idx': query_ent_idx, 'query_ent_mask': query_ent_mask,
                    'doc_ent_idx': doc_ent_idx, 'doc_ent_mask': doc_ent_mask,
                    'query_des_idx': query_des_idx, 'doc_des_idx': doc_des_idx}
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def __len__(self) -> int:
        return self._count
