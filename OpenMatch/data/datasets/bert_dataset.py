from typing import List, Tuple, Dict, Any

import json

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

class BertDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        tokenizer: AutoTokenizer,
        mode: str,
        query_max_len: int = 32,
        doc_max_len: int = 256,
        max_input: int = 1280000,
        task: str = 'ranking',
        template: str = None,
    ) -> None:
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._mode = mode
        self._query_max_len = query_max_len
        self._doc_max_len = doc_max_len
        self._seq_max_len = query_max_len + doc_max_len + 3
        self._max_input = max_input
        self._task = task
        if self._seq_max_len > 512:
            raise ValueError('query_max_len + doc_max_len + 3 > 512.')

        if self._task.startswith("prompt"):
            assert template is not None
            self._template = template
        # self._pos_word = pos_word
        # self._neg_word = neg_word
        # self._pos_word_id = tokenizer(pos_word, add_special_tokens=False)["input_ids"]
        # self._neg_word_id = tokenizer(neg_word, add_special_tokens=False)["input_ids"]
        # print(self._pos_word_id, self._neg_word_id)
        # if len(self._neg_word_id) > 1 or len(self._pos_word_id) > 1:
        #     raise ValueError("Label words longer than 1 after tokenization")
        # self._pos_word_id = self._pos_word_id[0]
        # self._neg_word_id = self._neg_word_id[0]

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
                    self._queries[line['query_id']] = line['query']
            self._docs = {}
            with open(self._dataset['docs'], 'r') as f:
                for line in f:
                    if self._dataset['docs'].split('.')[-1] == 'json' or self._dataset['docs'].split('.')[-1] == 'jsonl':
                        line = json.loads(line)
                    else:
                        doc_id, doc = line.strip('\n').split('\t')
                        line = {'doc_id': doc_id, 'doc': doc}
                    self._docs[line['doc_id']] = line['doc']
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
                            self._examples.append({'query_id': line[0], 'doc_id': line[1], 'label': int(line[2])})
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
                input_ids_pos = torch.tensor([item['input_ids_pos'] for item in batch])
                segment_ids_pos = torch.tensor([item['segment_ids_pos'] for item in batch])
                input_mask_pos = torch.tensor([item['input_mask_pos'] for item in batch])
                input_ids_neg = torch.tensor([item['input_ids_neg'] for item in batch])
                segment_ids_neg = torch.tensor([item['segment_ids_neg'] for item in batch])
                input_mask_neg = torch.tensor([item['input_mask_neg'] for item in batch])
                return {'input_ids_pos': input_ids_pos, 'segment_ids_pos': segment_ids_pos, 'input_mask_pos': input_mask_pos,
                        'input_ids_neg': input_ids_neg, 'segment_ids_neg': segment_ids_neg, 'input_mask_neg': input_mask_neg}
            elif self._task == 'classification':
                input_ids = torch.tensor([item['input_ids'] for item in batch])
                segment_ids = torch.tensor([item['token_type_ids'] for item in batch])
                input_mask = torch.tensor([item['attention_mask'] for item in batch])
                label = torch.tensor([item['label'] for item in batch])
                return {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask, 'label': label}
            elif self._task == "prompt_ranking":
                input_ids_pos = torch.tensor([item['input_ids_pos'] for item in batch])
                segment_ids_pos = torch.tensor([item['segment_ids_pos'] for item in batch])
                input_mask_pos = torch.tensor([item['input_mask_pos'] for item in batch])
                mask_pos_pos = torch.tensor([item['mask_pos_pos'] for item in batch])
                input_ids_neg = torch.tensor([item['input_ids_neg'] for item in batch])
                segment_ids_neg = torch.tensor([item['segment_ids_neg'] for item in batch])
                input_mask_neg = torch.tensor([item['input_mask_neg'] for item in batch])
                mask_pos_neg = torch.tensor([item['mask_pos_neg'] for item in batch])
                return {'input_ids_pos': input_ids_pos, 'segment_ids_pos': segment_ids_pos, 'input_mask_pos': input_mask_pos, "mask_pos_pos": mask_pos_pos, 
                        'input_ids_neg': input_ids_neg, 'segment_ids_neg': segment_ids_neg, 'input_mask_neg': input_mask_neg, "mask_pos_neg": mask_pos_neg}
            elif self._task == "prompt_classification":
                input_ids = torch.tensor([item['input_ids'] for item in batch])
                segment_ids = torch.tensor([item['token_type_ids'] for item in batch])
                mask_pos = torch.tensor([item['mask_pos'] for item in batch])
                input_mask = torch.tensor([item['attention_mask'] for item in batch])
                label = torch.tensor([item['label'] for item in batch])
                return {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask, "mask_pos": mask_pos, 'label': label}
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
        elif self._mode == 'dev':
            if self._task.startswith("prompt"):
                query_id = [item['query_id'] for item in batch]
                doc_id = [item['doc_id'] for item in batch]
                label = [item['label'] for item in batch]
                mask_pos = torch.tensor([item["mask_pos"] for item in batch])
                retrieval_score = torch.tensor([item['retrieval_score'] for item in batch])
                input_ids = torch.tensor([item['input_ids'] for item in batch])
                segment_ids = torch.tensor([item['token_type_ids'] for item in batch])
                input_mask = torch.tensor([item['attention_mask'] for item in batch])
                return {'query_id': query_id, 'doc_id': doc_id, 'label': label, 'retrieval_score': retrieval_score, "mask_pos": mask_pos, 
                        'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask}
            else:
                query_id = [item['query_id'] for item in batch]
                doc_id = [item['doc_id'] for item in batch]
                label = [item['label'] for item in batch]
                retrieval_score = torch.tensor([item['retrieval_score'] for item in batch])
                input_ids = torch.tensor([item['input_ids'] for item in batch])
                segment_ids = torch.tensor([item['token_type_ids'] for item in batch])
                input_mask = torch.tensor([item['attention_mask'] for item in batch])
                return {'query_id': query_id, 'doc_id': doc_id, 'label': label, 'retrieval_score': retrieval_score,
                        'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask}
        elif self._mode == 'test':
            if self._task.startswith("prompt"):
                query_id = [item['query_id'] for item in batch]
                doc_id = [item['doc_id'] for item in batch]
                mask_pos = torch.tensor([item["mask_pos"] for item in batch])
                retrieval_score = torch.tensor([item['retrieval_score'] for item in batch])
                input_ids = torch.tensor([item['input_ids'] for item in batch])
                segment_ids = torch.tensor([item['token_type_ids'] for item in batch])
                input_mask = torch.tensor([item['attention_mask'] for item in batch])
                return {'query_id': query_id, 'doc_id': doc_id, 'retrieval_score': retrieval_score, "mask_pos": mask_pos, 
                        'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask}
            else:
                query_id = [item['query_id'] for item in batch]
                doc_id = [item['doc_id'] for item in batch]
                retrieval_score = torch.tensor([item['retrieval_score'] for item in batch])
                input_ids = torch.tensor([item['input_ids'] for item in batch])
                segment_ids = torch.tensor([item['token_type_ids'] for item in batch])
                input_mask = torch.tensor([item['attention_mask'] for item in batch])
                return {'query_id': query_id, 'doc_id': doc_id, 'retrieval_score': retrieval_score,
                        'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask}
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def pack_bert_features(self, query_tokens: List[str], doc_tokens: List[str]):
        input_tokens = [self._tokenizer.cls_token] + query_tokens + [self._tokenizer.sep_token] + doc_tokens + [self._tokenizer.sep_token]
        input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
        segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_tokens) + 1)
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
        if self._id:
            example['query'] = self._queries[example['query_id']]
            if self._mode == 'train' and self._task == 'ranking':
                example['doc_pos'] = self._docs[example['doc_pos_id']]
                example['doc_neg'] = self._docs[example['doc_neg_id']]
            else:
                example['doc'] = self._docs[example['doc_id']]
        if self._mode == 'train':
            if self._task == 'ranking':
                query_tokens = self._tokenizer.tokenize(example['query'])[:self._query_max_len]
                doc_tokens_pos = self._tokenizer.tokenize(example['doc_pos'])[:self._seq_max_len-len(query_tokens)-3]
                doc_tokens_neg = self._tokenizer.tokenize(example['doc_neg'])[:self._seq_max_len-len(query_tokens)-3]

                input_ids_pos, input_mask_pos, segment_ids_pos = self.pack_bert_features(query_tokens, doc_tokens_pos)
                input_ids_neg, input_mask_neg, segment_ids_neg = self.pack_bert_features(query_tokens, doc_tokens_neg)
                return {'input_ids_pos': input_ids_pos, 'segment_ids_pos': segment_ids_pos, 'input_mask_pos': input_mask_pos,
                        'input_ids_neg': input_ids_neg, 'segment_ids_neg': segment_ids_neg, 'input_mask_neg': input_mask_neg}
            elif self._task == 'classification':
                tokenizer_output = self._tokenizer(example["query"], example["doc"], padding="max_length", truncation="only_second", max_length=512)
                output = {"label": example["label"]}
                output.update(tokenizer_output)
                return output
            elif self._task == "prompt_ranking":
                output = {}

                for doc_type in ["pos", "neg"]:
                    text = "'' " + example["query"] + " '' is [MASK] to '' " + example["doc_" + doc_type] + " ''"
                    input_ids = self._tokenizer.encode(text)[:512]
                    segment_ids = [0] * len(input_ids)
                    input_mask = [1] * len(input_ids)

                    padding_len = 512 - len(input_ids)
                    input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
                    input_mask = input_mask + [0] * padding_len
                    segment_ids = segment_ids + [0] * padding_len

                    assert len(input_ids) == len(input_mask) == len(segment_ids) == 512
                    mask_pos = input_ids.index(self._tokenizer.mask_token_id)
                    output.update({"input_ids_" + doc_type: input_ids, "segment_ids_" + doc_type: segment_ids, "input_mask_" + doc_type: input_mask, "mask_pos_" + doc_type: mask_pos})
                
                return output

            elif self._task == "prompt_classification":
                # text = "'' " + example["query"] + " '' is [MASK] to '' " + example["doc"] + " ''"
                doc = example["doc"].strip()
                text = self._template.replace("<q>", example["query"]).replace("<d>", doc)
                # print(text)
                input_ids = self._tokenizer.encode(text)[:512]
                segment_ids = [0] * len(input_ids)
                input_mask = [1] * len(input_ids)

                padding_len = 512 - len(input_ids)
                input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
                input_mask = input_mask + [0] * padding_len
                segment_ids = segment_ids + [0] * padding_len

                assert len(input_ids) == len(input_mask) == len(segment_ids) == 512
                mask_pos = input_ids.index(self._tokenizer.mask_token_id)
                # print(mask_pos)
                return {'input_ids': input_ids, 'token_type_ids': segment_ids, 'attention_mask': input_mask, "mask_pos": mask_pos, 'label': example['label']}

            else:
                raise ValueError('Task must be `ranking` or `classification`.')
        elif self._mode == 'dev':
            if self._task.startswith("prompt"):
                doc = example["doc"].strip()
                text = self._template.replace("<q>", example["query"]).replace("<d>", doc)
                input_ids = self._tokenizer.encode(text)[:512]
                segment_ids = [0] * len(input_ids)
                input_mask = [1] * len(input_ids)

                padding_len = 512 - len(input_ids)
                input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
                input_mask = input_mask + [0] * padding_len
                segment_ids = segment_ids + [0] * padding_len

                assert len(input_ids) == len(input_mask) == len(segment_ids) == 512
                mask_pos = input_ids.index(self._tokenizer.mask_token_id)
                return {"input_ids": input_ids, "token_type_ids": segment_ids, "attention_mask": input_mask, "mask_pos": mask_pos, 
                        'query_id': example['query_id'], 'doc_id': example['doc_id'], 'label': example['label'], 'retrieval_score': example['retrieval_score']}
            else:
                tokenizer_output = self._tokenizer(example["query"], example["doc"], padding="max_length", truncation="only_second", max_length=512)
                output = {'query_id': example['query_id'], 'doc_id': example['doc_id'], 'label': example['label'], 'retrieval_score': example['retrieval_score']}
                output.update(tokenizer_output)
                return output
        elif self._mode == 'test':
            if self._task.startswith("prompt"):
                doc = example["doc"].strip()
                text = self._template.replace("<q>", example["query"]).replace("<d>", doc)
                input_ids = self._tokenizer.encode(text)[:512]
                segment_ids = [0] * len(input_ids)
                input_mask = [1] * len(input_ids)

                padding_len = 512 - len(input_ids)
                input_ids = input_ids + [self._tokenizer.pad_token_id] * padding_len
                input_mask = input_mask + [0] * padding_len
                segment_ids = segment_ids + [0] * padding_len

                assert len(input_ids) == len(input_mask) == len(segment_ids) == 512
                mask_pos = input_ids.index(self._tokenizer.mask_token_id)
                return {"input_ids": input_ids, "token_type_ids": segment_ids, "attention_mask": input_mask, "mask_pos": mask_pos, 
                        'query_id': example['query_id'], 'doc_id': example['doc_id'], 'retrieval_score': example['retrieval_score']}
            else:
                tokenizer_output = self._tokenizer(example["query"], example["doc"], padding="max_length", truncation="only_second", max_length=512)
                output = {'query_id': example['query_id'], 'doc_id': example['doc_id'], 'retrieval_score': example['retrieval_score']}
                output.update(tokenizer_output)
                return output
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def __len__(self) -> int:
        return self._count
