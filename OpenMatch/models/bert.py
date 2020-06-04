from typing import Tuple

import torch
import torch.nn as nn

from transformers import AutoModel

class Bert(nn.Module):
    def __init__(
        self,
        pretrained: str,
        enc_dim: int = 768,
        task: str = 'ranking'
    ) -> None:
        super(Bert, self).__init__()
        self._pretrained = pretrained
        self._enc_dim = enc_dim
        self._task = task

        self._model = AutoModel.from_pretrained(self._pretrained)
        if self._task == 'ranking':
            self._dense = nn.Linear(self._enc_dim, 1)
        elif self._task == 'classification':
            self._dense = nn.Linear(self._enc_dim, 2)
        else:
            raise ValueError('Task must be `ranking` or `classification`.')

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor = None, segment_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self._model(input_ids, attention_mask = input_mask, token_type_ids = segment_ids)
        score = self._dense(output[0][:, 0, :]).squeeze(-1)
        return score, output[0][:, 0, :]
