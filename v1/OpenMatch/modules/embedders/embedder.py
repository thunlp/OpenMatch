from typing import List

import torch
import torch.nn as nn

class Embedder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        embed_matrix: List[float] = None
    ) -> None:
        super(Embedder, self).__init__()
        self._vocab_size = vocab_size
        self._embed_dim = embed_dim

        self._embedder = nn.Embedding(self._vocab_size, self._embed_dim, padding_idx=0)
        if embed_matrix is not None:
            self._embed_matrix = torch.tensor(embed_matrix)
            self._embedder.weight = nn.Parameter(self._embed_matrix, requires_grad=True)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        embed = self._embedder(idx)
        return embed
