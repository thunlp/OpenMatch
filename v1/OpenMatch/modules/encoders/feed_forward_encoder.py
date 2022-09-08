import torch
import torch.nn as nn

class FeedForwardEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ) -> None:
        super(FeedForwardEncoder, self).__init__()
        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim

        self._fc1 = torch.nn.Linear(self._embed_dim, self._hidden_dim)
        self._fc2 = torch.nn.Linear(self._hidden_dim, self._embed_dim)
        self._dropout = nn.Dropout(dropout)
        self._activation = nn.ReLU()
        self._norm = nn.LayerNorm(self._embed_dim)

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        enc = self._dropout(self._fc2(self._activation(self._fc1(embed))))
        enc = self._norm(embed + enc)
        return enc
