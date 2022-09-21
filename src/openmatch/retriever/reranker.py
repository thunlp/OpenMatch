import logging
import os
from contextlib import nullcontext
from typing import Dict

import torch
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.trainer_pt_utils import IterableDatasetShard

from ..arguments import InferenceArguments as EncodingArguments
from ..dataset import InferenceDataset, RRInferenceCollator
from ..modeling import RRModel
from ..utils import (load_from_trec, merge_retrieval_results_by_score,
                     save_as_trec)

logger = logging.getLogger(__name__)


def encode_pair(tokenizer, item1, item2, max_len_1=32, max_len_2=128):
    return tokenizer.encode_plus(
        item1 + item2,
        truncation='longest_first',
        padding='max_length',
        max_length=max_len_1 + max_len_2 + 2,
    )


def add_to_result_dict(result_dicts, qids, dids, scores):
    for qid, did, score in zip(qids, dids, scores):
        if qid not in result_dicts:
            result_dicts[qid] = {}
        result_dicts[qid][did] = float(score)


class RRPredictDataset(IterableDataset):

    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        query_dataset: InferenceDataset, 
        corpus_dataset: InferenceDataset, 
        run: Dict[str, Dict[str, float]]
    ):
        super(RRPredictDataset, self).__init__()
        self.tokenizer = tokenizer
        self.query_dataset = query_dataset
        self.corpus_dataset = corpus_dataset
        self.run = run

    def __iter__(self):
        def gen_q_d_pair():
            for qid, did_and_scores in self.run.items():
                for did, _ in did_and_scores.items():
                    yield {
                        "query_id": qid, 
                        "doc_id": did, 
                        **encode_pair(
                            self.tokenizer, 
                            self.query_dataset[qid]["input_ids"], 
                            self.corpus_dataset[did]["input_ids"], 
                            self.query_dataset.max_len, 
                            self.corpus_dataset.max_len
                        ),
                    }
        return gen_q_d_pair()


class Reranker:

    def __init__(
        self, 
        model: RRModel, 
        tokenizer: PreTrainedTokenizer, 
        corpus_dataset: Dataset, 
        args: EncodingArguments
    ):
        logger.info("Initializing reranker")
        self.model = model
        self.tokenizer = tokenizer
        self.corpus_dataset = corpus_dataset
        self.args = args

        self.model = model.to(self.args.device)
        self.model.eval()

    def rerank(self, query_dataset: InferenceDataset, run: Dict[str, Dict[str, float]]):
        return_dict = {}
        dataset = RRPredictDataset(self.tokenizer, query_dataset, self.corpus_dataset, run)
        if self.args.world_size > 1:
            dataset = IterableDatasetShard(
                dataset,
                batch_size=self.args.per_device_eval_batch_size,
                drop_last=False,
                num_processes=self.args.world_size,
                process_index=self.args.process_index
            )
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=RRInferenceCollator(),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        with torch.no_grad():
            for qids, dids, batch in tqdm(dataloader, desc="Reranking", disable=self.args.local_process_index > 0):
                with amp.autocast() if self.args.fp16 else nullcontext():
                    for k, v in batch.items():
                        batch[k] = v.to(self.args.device)
                    outputs = self.model.encode(batch)
                if len(outputs.shape) == 2 and outputs.shape[1] == 2:
                    outputs = F.log_softmax(outputs, dim=1)[:, 1]
                scores = outputs.cpu().numpy()
                add_to_result_dict(return_dict, qids, dids, scores)
    
        if self.args.world_size > 1:
            save_as_trec(return_dict, self.args.trec_save_path + ".rank.{}".format(self.args.process_index))
            torch.distributed.barrier()
            if self.args.process_index == 0:
                # aggregate results
                all_results = []
                for i in range(self.args.world_size):
                    all_results.append(load_from_trec(self.args.trec_save_path + ".rank.{}".format(i)))
                return_dict = merge_retrieval_results_by_score(all_results)
                # remove temp files
                for i in range(self.args.world_size):
                    os.remove(self.args.trec_save_path + ".rank.{}".format(i))
            torch.distributed.barrier()

        return return_dict
