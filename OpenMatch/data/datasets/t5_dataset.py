from os import truncate
from typing import List, Tuple, Dict, Any

import json

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

class t5Dataset(Dataset):
    def __init__(
        self,
        dataset: str,
        tokenizer: AutoTokenizer,
        mode: str,
        query_max_len: int = 25,
        doc_max_len: int = 250,
        max_input: int = 1280000,
        task: str = 'classification',
        isv11: bool=False
    ) -> None:
        self._label_mapping=[6136,1176]
        self._dataset = dataset
        self._tokenizer = tokenizer
        #self._tokenizer.sep_token='</s>'
        self._mode = mode
        self._query_max_len = query_max_len
        self._doc_max_len = doc_max_len
        self._seq_max_len = query_max_len + doc_max_len + 11
        self._max_input = max_input
        self._task = task
        self.isv11=isv11
        if self._seq_max_len > 384:
            raise ValueError('query_max_len + doc_max_len + 11 > 384.')

        
        if isinstance(self._dataset, str):
            with open(self._dataset, 'r') as f:
                self._examples = []
                for i, line in enumerate(f):
                    if i >= self._max_input:
                        break
                    if self._mode != 'train' or self._dataset.split('.')[-1] == 'json' or self._dataset.split('.')[-1] == 'jsonl':
                        line = json.loads(line)
                    else:
                        if self._task == 'classification':
                            query, doc, label = line.strip('\n').split('\t')
                            line = {'query': query, 'doc': doc, 'label': int(label)}
                        else:
                            raise ValueError('Task must be `classification`.')
                    self._examples.append(line)
        else:
            raise ValueError('Dataset must be `str`.')
        self._count = len(self._examples)
        

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        #print(example['query'].split(' '))
        #print(len(example['query'].split(' ')))
        #exit(0)
        if len(example["query"].split(' ')) > self._query_max_len:
            example['query']=' '.join(example['query'].split(' ')[:self._query_max_len])
        if len(example["doc"].split(' ')) > self._doc_max_len:
            example['doc']=' '.join(example['doc'].split(' ')[:self._doc_max_len])
        if self._mode == 'train':
            if self._task == 'classification':
                text='Query: '+example["query"]+' Document: '+example["doc"]+' Relevant: '
                #text=example['query']+' </s> '+example['doc']
                #label_text=self._label_mapping[example['label']]
                #label_text="<pad>"
                output=self._tokenizer(text,padding="max_length",truncation=True,max_length=384)
                #label=self._tokenizer(label_text).input_ids
                if self.isv11:
                    label=self._label_mapping[example['label']]
                else:
                    label=example['label']
                output.update({'labels':[0],'label':label})
                return output
            else:
                raise ValueError('Task must be  `classification`.')
        elif self._mode == 'dev':
            text='Query: '+example["query"]+' Document: '+example["doc"]+' Revelant: '
            #text=example['query']+' </s> '+example['doc']
            #label_text=self._label_mapping[example['label']]
            #label_text="<pad>"
            input_ids=self._tokenizer(text,padding="max_length",truncation=True,max_length=384)
            #label=self._tokenizer(label_text).input_ids
            
            input_ids.update({"labels":[0]})
            #tokenizer_output = self._tokenizer(example["query"], example["doc"], padding="max_length", truncation="only_second", max_length=384)
            output = {'query_id': example['query_id'], 'doc_id': example['doc_id']}
            output.update(input_ids)
            #output : [qid,did,label,label,rs,attention_masks,input_ids,labels]
            if self.isv11:
                label=self._label_mapping[example['label']]
            else:
                label=example['label']
            output.update({'label':label})
            return output
        elif self._mode == 'test':
            text='Query: '+example["query"]+' Document: '+example["doc"]+' Revelant: '
            #text=example['query']+' </s> '+example['doc']
            #label_text=self._label_mapping[example['label']]
            #label_text="<pad>"
            input_ids=self._tokenizer(text,padding="max_length",truncation=True,max_length=384)
            #label=self._tokenizer(label_text).input_ids
            
            input_ids.update({"labels":[0]})
            #tokenizer_output = self._tokenizer(example["query"], example["doc"], padding="max_length", truncation="only_second", max_length=384)
            output = {'query_id': example['query_id'], 'doc_id': example['doc_id']}
            output.update(input_ids)
            #output : [qid,did,label,label,rs,attention_masks,input_ids,labels]
            if self.isv11:
                label=self._label_mapping[example['label']]
            else:
                label=example['label']
            output.update({'label':label})
            return output
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def __len__(self) -> int:
        return self._count

    def collate(self, batch: Dict[str, Any]):
        if self._mode == 'train':
            if self._task == 'classification':
                input_ids = torch.tensor([item['input_ids'] for item in batch])
                labels = torch.tensor([item['labels'] for item in batch])
                attention_mask = torch.tensor([item['attention_mask'] for item in batch])
                label = torch.tensor([item['label'] for item in batch])
                
                return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask, 'label': label}
            else:
                raise ValueError('Task must be  `classification`.')
        elif self._mode == 'dev':  
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            label = torch.tensor([item['label'] for item in batch])
            
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            labels = torch.tensor([item['labels'] for item in batch])
            attention_mask = torch.tensor([item['attention_mask'] for item in batch])
            return {'query_id': query_id, 'doc_id': doc_id, 'label': label,
                    'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}
        elif self._mode == 'test':
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            label = torch.tensor([item['label'] for item in batch])
            
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            labels = torch.tensor([item['labels'] for item in batch])
            attention_mask = torch.tensor([item['attention_mask'] for item in batch])
            return {'query_id': query_id, 'doc_id': doc_id, 'label': label,
                    'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')