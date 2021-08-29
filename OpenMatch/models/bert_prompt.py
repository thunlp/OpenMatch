from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, AutoModel, AutoModelForMaskedLM, AutoConfig

class BertPrompt(nn.Module):
    def __init__(
        self,
        pretrained: str,
        mode: str = 'cls',
        task: str = 'ranking',
        pos_word_id: int = 0,
        neg_word_id: int = 0
    ) -> None:
        super(BertPrompt, self).__init__()
        self._pretrained = pretrained
        self._mode = mode
        self._task = task

        self._config = AutoConfig.from_pretrained(self._pretrained)
        self._model = AutoModelForMaskedLM.from_pretrained(self._pretrained, config=self._config)
        self._pos_word_id = pos_word_id
        self._neg_word_id = neg_word_id

        # if self._task == 'ranking':
        #     # self._dense = nn.Linear(self._config.hidden_size, 1)
        #     pass
        # elif self._task == 'classification':
        #     # self._dense = nn.Linear(self._config.hidden_size, 2)
        #     pass
        # else:
        #     raise ValueError('Task must be `ranking` or `classification`.')

    def forward(self, input_ids: torch.Tensor, masked_token_pos: torch.Tensor, input_mask: torch.Tensor = None, segment_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self._model(input_ids, attention_mask = input_mask, token_type_ids = segment_ids)[0]
        if self._mode == 'cls':
            # logits = output[0][:, 0, :]
            pass
        elif self._mode == 'pooling':
            # logits = output[1]
            pass
        else:
            raise ValueError('Mode must be `cls` or `pooling`.')
        
        # print(masked_token_pos)

        vocab_size = output.shape[2]
        masked_token_pos = torch.unsqueeze(masked_token_pos, 1)
        masked_token_pos = torch.unsqueeze(masked_token_pos, 2)
        masked_token_pos = torch.stack([masked_token_pos] * vocab_size, 2)
        masked_token_pos = torch.squeeze(masked_token_pos, 3)
        masked_token_logits = torch.gather(output, 1, masked_token_pos).squeeze()  # batch_size * vocab_size
        rel_and_irrel_logits = masked_token_logits[:, [self._neg_word_id, self._pos_word_id]]
        # print()
        # rel_and_irrel_logits = F.softmax(rel_and_irrel_logits, dim=1)
        # print(rel_and_irrel_logits)

        return rel_and_irrel_logits, masked_token_logits
