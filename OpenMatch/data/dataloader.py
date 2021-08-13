from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from OpenMatch.data.datasets import Dataset

class DataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: str = False,
        num_workers: int = 0,
        sampler = None,
    ) -> None:
        super().__init__(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers,
            collate_fn = dataset.collate,
            sampler = sampler,
        )
