from typing import Tuple

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel

class Bert(nn.Module):
    def __init__(
        self,
        pretrained: str,
        task: str = 'ranking'
    ) -> None:
        super(Bert, self).__init__()
        self._pretrained = pretrained
        self._task = task

        self._config = AutoConfig.from_pretrained(self._pretrained)
        self._model = AutoModel.from_pretrained(self._pretrained, config=self._config)
        if self._task == 'ranking':
            self._dense = nn.Linear(self._config.hidden_size, 1)
        elif self._task == 'classification':
            self._dense = nn.Linear(self._config.hidden_size, 2)
        else:
            raise ValueError('Task must be `ranking` or `classification`.')

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor = None, segment_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self._model(input_ids, attention_mask = input_mask, token_type_ids = segment_ids)
        score = self._dense(output[0][:, 0, :]).squeeze(-1)
        return score, output[0][:, 0, :]
