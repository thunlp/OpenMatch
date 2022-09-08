import torch
from torch import nn
import torch.distributed as dist
torch.multiprocessing.set_sharing_strategy('file_system')
# from multiprocessing import Process
from torch.utils.data import DataLoader, Sampler, Dataset, TensorDataset, IterableDataset
import math
import logging

import random
import numpy as np

import glob
import os
from collections import OrderedDict
import json

def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def merge_resfile(split_pattern, output_file):
    splits = glob.glob(split_pattern)
    logging.info("temperary validation trec files: {}".format(splits))
    res_dict = {}
    for s in splits:
        with open(s,'r') as f:
            for line in f:
                qid, _, pid, _, score, _ = line.strip().split() # ranking is meaningless in distributed inference
                if qid not in res_dict:
                    res_dict[qid]=[(pid,score)]
                else:
                    res_dict[qid].append((pid,score))
        os.remove(s)
    cnt = 0
    with open(output_file,'w') as f:
        for qid in res_dict:
            res_dict[qid] = sorted(res_dict[qid], key=lambda x: x[1], reverse=True)
            rank = 1 # start from 1
            for pid, score in res_dict[qid]:
                f.write(qid+' Q0 '+ str(pid) +' '+str(rank)+' '+ str(score) +' openmatch\n')
                rank+=1
                cnt+=1
    logging.info("merge total {} lines".format(cnt))

def merge_featfile(split_pattern, jsonl_file, output_file):
    qid_list = []
    qid_to_pid_by_order = {}
    with open(jsonl_file,'r') as f:
        for line in f:
            line_dict = json.loads(line)
            qid = line_dict['query_id']
            pid = line_dict['doc_id']
            if qid not in qid_list:
                qid_list.append(qid)
                qid_to_pid_by_order[qid]=[pid]
            else:
                qid_to_pid_by_order[qid].append(pid)

    splits = glob.glob(split_pattern)
    logging.info("temperary validation trec files: {}".format(splits))
    feat_dict={}
    for s in splits:
        with open(s,'r') as f:
            for line in f:
                line_arr = line.strip().split()
                qid=line_arr[1].replace("id:","")
                pid=line_arr[2].replace("doc_id:","")
                del line_arr[2]
                new_line = " ".join(line_arr) + "\n"
                if qid not in feat_dict:
                    feat_dict[qid]={pid:new_line}
                else:
                    feat_dict[qid][pid]=new_line
        os.remove(s)
    cnt = 0
    with open(output_file,'w') as f:
        for qid in qid_list:
            for pid in qid_to_pid_by_order[qid]:
                line = feat_dict[qid][pid]
                f.write(line)
                cnt+=1

    logging.info("merge total {} lines of features".format(cnt))

# https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py
class DistributedEvalSampler(Sampler):
    r"""
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default
    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.
    Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    Example::
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)             # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch


def set_dist_args(args):
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see
        # https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(
                args.server_ip,
                args.server_port),
            redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
   
    # Set seed
    seed=42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # store args for multi process
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()
    # assign args.world_size
    else:
        args.world_size = 1

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logging.warning(
        # "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        "Process gpu rank: %s, process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        args.rank if (args.local_rank != -1) else 0,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        # args.fp16,
    )


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

# create new OrderedDict that does not contain `module.`
def clean_dict_name(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict