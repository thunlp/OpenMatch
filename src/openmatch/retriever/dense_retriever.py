import gc
import glob
import os
import pickle
from contextlib import nullcontext
from typing import Dict, List
import logging

import faiss
import numpy as np
import torch
from torch.cuda import amp
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers.trainer_pt_utils import IterableDatasetShard

from ..arguments import InferenceArguments as EncodingArguments
from ..dataset import DRInferenceCollator
from ..modeling import DRModelForInference, DROutput
from ..utils import merge_retrieval_results_by_score

logger = logging.getLogger(__name__)


class Retriever:

    def __init__(self, model: DRModelForInference, corpus_dataset: IterableDataset, args: EncodingArguments):
        logger.info("Initializing retriever")
        self.model = model
        self.corpus_dataset = corpus_dataset
        self.args = args
        self.doc_lookup = []
        self.query_lookup = []

        self.model.to(self.args.device)
        self.model.eval()

    def _initialize_faiss_index(self, dim: int):
        self.index = None
        cpu_index = faiss.IndexFlatIP(dim)
        self.index = cpu_index

    def _move_index_to_gpu(self):
        logger.info("Moving index to GPU(s)")
        ngpu = faiss.get_num_gpus()
        gpu_resources = []
        for i in range(ngpu):
            res = faiss.StandardGpuResources()
            gpu_resources.append(res)
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.usePrecomputed = False
        vres = faiss.GpuResourcesVector()
        vdev = faiss.Int32Vector()
        for i in range(0, ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        self.index = faiss.index_cpu_to_gpu_multiple(vres, vdev, self.index, co)

    def doc_embedding_inference(self):
        # Note: during evaluation, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if self.corpus_dataset is None:
            raise ValueError("No corpus dataset provided")
        dataloader = DataLoader(
            self.corpus_dataset,
            batch_size=self.args.per_device_eval_batch_size,  # Note that we do not support DataParallel here
            collate_fn=DRInferenceCollator(),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        encoded = []
        lookup_indices = []
        for (batch_ids, batch) in tqdm(dataloader, disable=self.args.local_process_index > 0):
            lookup_indices.extend(batch_ids)
            with amp.autocast() if self.args.fp16 else nullcontext():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(self.args.device)
                    model_output: DROutput = self.model(passage=batch)
                    encoded.append(model_output.p_reps.cpu().detach().numpy())
        encoded = np.concatenate(encoded)

        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(os.path.join(self.args.output_dir, "embeddings.corpus.rank.{}".format(self.args.process_index)), 'wb') as f:
            pickle.dump((encoded, lookup_indices), f, protocol=4)

        del encoded
        del lookup_indices
        
        if self.args.world_size > 1:
            torch.distributed.barrier()

    def init_index_and_add(self, partition: str = None):
        logger.info("Initializing Faiss index from pre-computed document embeddings")
        partitions = [partition] if partition is not None else glob.glob(os.path.join(self.args.output_dir, "embeddings.corpus.rank.*"))
        for i, part in enumerate(partitions):
            with open(part, 'rb') as f:
                data = pickle.load(f)
            encoded = data[0]
            lookup_indices = data[1]
            if i == 0:
                dim = encoded.shape[1]
                self._initialize_faiss_index(dim)
            self.index.add(encoded)
            self.doc_lookup.extend(lookup_indices)

    @classmethod
    def build_all(cls, model: DRModelForInference, corpus_dataset: IterableDataset, args: EncodingArguments):
        retriever = cls(model, corpus_dataset, args)
        retriever.doc_embedding_inference()
        if args.process_index == 0:
            retriever.init_index_and_add()
        if args.world_size > 1:
            torch.distributed.barrier()
        return retriever

    @classmethod
    def build_embeddings(cls, model: DRModelForInference, corpus_dataset: IterableDataset, args: EncodingArguments):
        retriever = cls(model, corpus_dataset, args)
        retriever.doc_embedding_inference()
        return retriever

    @classmethod
    def from_embeddings(cls, model: DRModelForInference, args: EncodingArguments):
        retriever = cls(model, None, args)
        if args.process_index == 0:
            retriever.init_index_and_add()
        if args.world_size > 1:
            torch.distributed.barrier()
        return retriever

    def reset_index(self):
        if self.index:
            self.index.reset()
        self.doc_lookup = []
        self.query_lookup = []

    def query_embedding_inference(self, query_dataset: IterableDataset):
        dataloader = DataLoader(
            query_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=DRInferenceCollator(),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        encoded = []
        lookup_indices = []
        for (batch_ids, batch) in tqdm(dataloader, disable=self.args.local_process_index > 0):
            lookup_indices.extend(batch_ids)
            with amp.autocast() if self.args.fp16 else nullcontext():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(self.args.device)
                    model_output: DROutput = self.model(query=batch)
                    encoded.append(model_output.q_reps.cpu().detach().numpy())
        
        encoded = np.concatenate(encoded)

        with open(os.path.join(self.args.output_dir, "embeddings.query.rank.{}".format(self.args.process_index)), 'wb') as f:
            pickle.dump((encoded, lookup_indices), f, protocol=4)
        
        if self.args.world_size > 1:
            torch.distributed.barrier()

    def search(self, topk: int = 100):
        logger.info("Searching")
        if self.index is None:
            raise ValueError("Index is not initialized")
        encoded = []
        for i in range(self.args.world_size):
            with open(os.path.join(self.args.output_dir, "embeddings.query.rank.{}".format(i)), 'rb') as f:
                data = pickle.load(f)
            lookup_indices = data[1]
            encoded.append(data[0])
            self.query_lookup.extend(lookup_indices)
        encoded = np.concatenate(encoded)

        return_dict = {}
        D, I = self.index.search(encoded, topk)
        original_indices = np.array(self.doc_lookup)[I]
        q = 0
        for scores_per_q, doc_indices_per_q in zip(D, original_indices):
            qid = str(self.query_lookup[q])
            return_dict[qid] = {}
            for doc_index, score in zip(doc_indices_per_q, scores_per_q):
                return_dict[qid][str(doc_index)] = float(score)
            q += 1

        logger.info("End searching with {} queries".format(len(return_dict)))

        return return_dict

    def retrieve(self, query_dataset: IterableDataset, topk: int = 100):
        self.query_embedding_inference(query_dataset)
        self.model.cpu()
        del self.model
        torch.cuda.empty_cache()
        results = {}
        if self.args.process_index == 0:
            if self.args.use_gpu:
                self._move_index_to_gpu()
            results = self.search(topk)
        if self.args.world_size > 1:
            torch.distributed.barrier()
        return results


class SuccessiveRetriever(Retriever):

    def __init__(self, model: DRModelForInference, corpus_dataset: IterableDataset, args: EncodingArguments):
        super().__init__(model, corpus_dataset, args)

    @classmethod
    def from_embeddings(cls, model: DRModelForInference, args: EncodingArguments):
        retriever = cls(model, None, args)
        return retriever

    def retrieve(self, query_dataset: IterableDataset, topk: int = 100):
        self.query_embedding_inference(query_dataset)
        del self.model
        torch.cuda.empty_cache()
        final_result = {}
        if self.args.process_index == 0:
            all_partitions = glob.glob(os.path.join(self.args.output_dir, "embeddings.corpus.rank.*"))
            for partition in all_partitions:
                logger.info("Loading partition {}".format(partition))
                self.init_index_and_add(partition)
                if self.args.use_gpu:
                    self._move_index_to_gpu()
                cur_result = self.search(topk)
                self.reset_index()
                final_result = merge_retrieval_results_by_score([final_result, cur_result], topk)
        if self.args.world_size > 1:
            torch.distributed.barrier()
        return final_result
