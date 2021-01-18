from typing import Tuple

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel

class BertMaxP(nn.Module):
    def __init__(
        self,
        pretrained: str,
        max_query_len: int,
        max_doc_len: int,
        mode: str = 'cls',
        task: str = 'ranking'
    ) -> None:
        super(BertMaxP, self).__init__()
        self._pretrained = pretrained
        self._max_query_len = max_query_len
        self._max_doc_len = max_doc_len
        self._mode = mode
        self._task = task

        self._config = AutoConfig.from_pretrained(self._pretrained)
        self._model = AutoModel.from_pretrained(self._pretrained, config=self._config)
        for p in self.parameters():
            p.requires_grad=False

        self._dense1 = nn.Linear(self._config.hidden_size, 128)
        self._activation = nn.ReLU()

        if self._task == 'ranking':
            self._dense2 = nn.Linear(128, 1)
        elif self._task == 'classification':
            self._dense2 = nn.Linear(128, 2)
        else:
            raise ValueError('Task must be `ranking` or `classification`.')

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor = None, segment_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        num = input_ids.size()[0]
        output = self._model(input_ids.view(num*4, self._max_query_len+self._max_doc_len+3), attention_mask = input_mask.view(num*4, self._max_query_len+self._max_doc_len+3), token_type_ids = segment_ids.view(num*4, self._max_query_len+self._max_doc_len+3))

        if self._mode == 'cls':
            logits = output[0][:, 0, :].view(num,4,-1).max(dim=1)[0]
        elif self._mode == 'pooling':
            logits = output[1].view(num,4,-1).max(dim=1)[0]
        else:
            raise ValueError('Mode must be `cls` or `pooling`.')
        logits = self._activation(self._dense1(logits))
        score = self._dense2(logits).squeeze(-1)
        return score, logits
