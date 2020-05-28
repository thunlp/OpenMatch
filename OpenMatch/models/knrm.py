from typing import List, Tuple

import torch
import torch.nn as nn

from OpenMatch.modules.embedders import Embedder
from OpenMatch.modules.matchers import KernelMatcher

class KNRM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        kernel_num: int = 21,
        embed_matrix: List[float] = None,
        task: str = 'ranking'
    ) -> None:
        super(KNRM, self).__init__()
        self._vocab_size = vocab_size
        self._embed_dim = embed_dim
        self._kernel_num = kernel_num
        self._embed_matrix = embed_matrix
        self._task = task

        self._embedder = Embedder(self._vocab_size, self._embed_dim, self._embed_matrix)
        self._matcher = KernelMatcher(self._embed_dim, self._kernel_num)
        if self._task == 'ranking':
            self._dense = nn.Linear(self._kernel_num, 1)
        elif self._task == 'classification':
            self._dense = nn.Linear(self._kernel_num, 2)
        else:
            raise ValueError('Task must be `ranking` or `classification`.')

    def forward(self, query_ids: torch.Tensor, query_masks: torch.Tensor, doc_ids: torch.Tensor, doc_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query_embed = self._embedder(query_ids)
        doc_embed = self._embedder(doc_ids)

        logits = self._matcher(query_embed, query_masks, doc_embed, doc_masks)
        score = self._dense(logits).squeeze(-1)
        return score, logits
