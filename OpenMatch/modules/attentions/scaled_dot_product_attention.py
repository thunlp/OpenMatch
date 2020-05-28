import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        dropout: float = 0.0
    ) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self._dropout = nn.Dropout(dropout)
        self._softmax = nn.Softmax(dim=2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float=None, attn_mask: torch.Tensor=None) -> torch.Tensor:
        attn = torch.bmm(query, key.transpose(1, 2))
        if scale is not None:
            attn *= scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, -1.0e32)
        attn = self._softmax(attn)
        attn = self._dropout(attn)
        context = torch.bmm(attn, value)
        return context, attn
