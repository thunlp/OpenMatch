import torch
import torch.nn as nn

from .scaled_dot_product_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        head_num: int = 8,
        dropout: float = 0.0
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self._embed_dim = embed_dim
        self._head_num = head_num
        self._head_dim = self._embed_dim // self._head_num
        assert self._head_dim * self._head_num == self._embed_dim, 'embed_dim must be divisible by num_heads'

        self._fcq = nn.Linear(self._embed_dim, self._head_dim * self._head_num)
        self._fck = nn.Linear(self._embed_dim, self._head_dim * self._head_num)
        self._fcv = nn.Linear(self._embed_dim, self._head_dim * self._head_num)
        self._attention = ScaledDotProductAttention(dropout)
        self._fc = nn.Linear(self._embed_dim, self._embed_dim)
        self._dropout = nn.Dropout(dropout)
        self._norm = nn.LayerNorm(self._embed_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor=None) -> torch.Tensor:
        residual = query
        batch_size = query.size(0)
        query = self._fcq(query).view(batch_size * self._head_num, -1, self._head_dim)
        key = self._fck(key).view(batch_size * self._head_num, -1, self._head_dim)
        value = self._fcv(value).view(batch_size * self._head_num, -1, self._head_dim)

        scale = (query.size(-1) // self._head_num) ** -0.5
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self._head_num, 1, 1)
        context, attn = self._attention(query, key, value, scale, attn_mask)
        context = context.view(batch_size, -1, self._head_num * self._head_dim)
        output = self._fc(context)
        output = self._dropout(output)
        output = self._norm(residual + output)
        return output, attn
