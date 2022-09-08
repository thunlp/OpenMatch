from typing import List, Tuple

import torch
import torch.nn as nn

from OpenMatch.modules.embedders import Embedder
from OpenMatch.modules.encoders import Conv1DEncoder
from OpenMatch.modules.matchers import KernelMatcher

class ConvKNRM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        kernel_num: int = 21,
        kernel_dim: int = 128,
        kernel_sizes: List[int] = [1, 2, 3],
        embed_matrix: List[float] = None,
        task: str = 'ranking'
    ) -> None:
        super(ConvKNRM, self).__init__()
        self._vocab_size = vocab_size
        self._embed_dim = embed_dim
        self._kernel_num = kernel_num
        self._kernel_dim = kernel_dim
        self._kernel_sizes = kernel_sizes
        self._embed_matrix = embed_matrix
        self._task = task

        self._embedder = Embedder(self._vocab_size, self._embed_dim, self._embed_matrix)
        self._encoder = Conv1DEncoder(self._embed_dim, self._kernel_dim, self._kernel_sizes)
        self._matcher = KernelMatcher(self._encoder.get_output_dim(), self._kernel_num)
        if self._task == 'ranking':
            self._dense = nn.Linear(self._kernel_num * (len(self._kernel_sizes) ** 2), 1)
        elif self._task == 'classification':
            self._dense = nn.Linear(self._kernel_num * (len(self._kernel_sizes) ** 2), 2)
        else:
            raise ValueError('Task must be `ranking` or `classification`.')

    def forward(self, query_ids: torch.Tensor, query_masks: torch.Tensor, doc_ids: torch.Tensor, doc_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query_embed = self._embedder(query_ids)
        doc_embed = self._embedder(doc_ids)
        _, query_encs = self._encoder(query_embed, query_masks)
        _, doc_encs = self._encoder(doc_embed, doc_masks)

        logits = torch.cat([self._matcher(query_enc, query_masks[:, :query_enc.size()[1]], doc_enc, doc_masks[:, :doc_enc.size()[1]])
                  for query_enc in query_encs for doc_enc in doc_encs], dim=1)
        score = self._dense(logits).squeeze(-1)
        return score, logits
