from typing import List, Tuple

import torch
import torch.nn as nn

from OpenMatch.modules.embedders import Embedder
from OpenMatch.modules.encoders import TransformerEncoder
from OpenMatch.modules.matchers import KernelMatcher

class TK(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        head_num: int = 10,
        hidden_dim: int = 100,
        layer_num: int = 2,
        kernel_num: int = 21,
        dropout: float = 0.0,
        embed_matrix: List[float] = None,
        task: str = 'ranking'
    ) -> None:
        super(TK, self).__init__()
        self._vocab_size = vocab_size
        self._embed_dim = embed_dim
        self._head_num = head_num
        self._hidden_dim = hidden_dim
        self._layer_num = layer_num
        self._kernel_num = kernel_num
        self._dropout = dropout
        self._embed_matrix = embed_matrix
        self._task = task

        self._embedder = Embedder(self._vocab_size, self._embed_dim, self._embed_matrix)
        self._encoder = TransformerEncoder(self._embed_dim, self._head_num, self._hidden_dim, self._layer_num, self._dropout)
        self._mixer = nn.Parameter(torch.full([1,1,1], 0.5, dtype=torch.float32, requires_grad=True))
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
        query_context = self._encoder(query_embed, ~query_masks.bool().unsqueeze(1).expand(-1, query_masks.size(1), -1))
        doc_context = self._encoder(doc_embed, ~doc_masks.bool().unsqueeze(1).expand(-1, doc_masks.size(1), -1))
        query_embed = (self._mixer * query_embed + (1 - self._mixer) * query_context)
        doc_embed = (self._mixer * doc_embed + (1 - self._mixer) * doc_context)

        logits = self._matcher(query_embed, query_masks, doc_embed, doc_masks)
        score = self._dense(logits).squeeze(-1)
        return score, logits
