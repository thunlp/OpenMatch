import torch
import torch.nn as nn

from OpenMatch.modules.attentions import MultiHeadAttention
from .feed_forward_encoder import FeedForwardEncoder
from .positional_encoder import PositionalEncoder

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,   
        embed_dim: int,
        head_num: int = 8,
        hidden_dim: int = 2048,
        dropout: float = 0.0
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self._embed_dim = embed_dim
        self._head_num = head_num
        self._hidden_dim = hidden_dim
        self._dropout = dropout

        self._attention = MultiHeadAttention(self._embed_dim, self._head_num, dropout=self._dropout)
        self._feed_forward = FeedForwardEncoder(self._embed_dim, self._hidden_dim, self._dropout)

    def forward(self, embed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embed, weights = self._attention(embed, embed, embed, attn_mask=mask)
        enc = self._feed_forward(embed)
        return enc

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_num: int = 8,
        hidden_dim: int = 2048,
        layer_num: int = 6,
        dropout: float = 0.0
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self._embed_dim = embed_dim
        self._head_num = head_num
        self._hidden_dim = hidden_dim
        self._layer_num = layer_num
        self._dropout = dropout

        self._pos_encoder = PositionalEncoder(self._embed_dim)
        self._layers = nn.ModuleList([
            TransformerEncoderLayer(self._embed_dim, self._head_num, self._hidden_dim, self._dropout) for _ in range(self._layer_num)
        ])

    def forward(self, embed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        enc = self._pos_encoder(embed)
        for layer in self._layers:
            enc = layer(enc, mask)
        return enc
