from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

from transformers import BertConfig, AutoModel, AutoModelForMaskedLM, AutoConfig

class BertPrompt(nn.Module):
    def __init__(
        self,
        pretrained: str,
        mode: str = 'cls',
        task: str = 'ranking',
        pos_word_id: int = 0,
        neg_word_id: int = 0,
        soft_prompt: bool = False
    ) -> None:
        super(BertPrompt, self).__init__()
        self._pretrained = pretrained
        self._mode = mode
        self._task = task

        self._config = AutoConfig.from_pretrained(self._pretrained)
        self._model = AutoModelForMaskedLM.from_pretrained(self._pretrained, config=self._config)
        self._pos_word_id = pos_word_id
        self._neg_word_id = neg_word_id
        self._soft_prompt = soft_prompt
        self.soft_embedding = None

        if self._soft_prompt:
            print("soft prompt")

            self.soft_embedding = nn.Embedding(100, self._config.hidden_size)
            self.model_embedding = self._model.get_input_embeddings()
            self.soft_embedding.weight.data = self.model_embedding.weight.data[:100, :].clone().detach().requires_grad_(True)
            for param in self._model.parameters():
                param.requires_grad_(False)
            self.new_lstm_head = nn.LSTM(
                input_size = self._config.hidden_size,
                hidden_size = self._config.hidden_size, # TODO P-tuning different in LAMA & FewGLUE
                # TODO dropout different in LAMA and FewGLUE
                num_layers=2,
                bidirectional=True,
                batch_first=True
            )
            self.new_mlp_head = nn.Sequential(
                nn.Linear(2*self._config.hidden_size, self._config.hidden_size),
                nn.ReLU(),
                nn.Linear(self._config.hidden_size, self._config.hidden_size)
            )
        # if fix_parameters:
        #     self._model.eval()

        # if self._task == 'ranking':
        #     # self._dense = nn.Linear(self._config.hidden_size, 1)
        #     pass
        # elif self._task == 'classification':
        #     # self._dense = nn.Linear(self._config.hidden_size, 2)
        #     pass
        # else:
        #     raise ValueError('Task must be `ranking` or `classification`.')

    def forward(self, input_ids: torch.Tensor, masked_token_pos: torch.Tensor, input_mask: torch.Tensor = None, segment_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # if self._fix_para:
        #     with torch.no_grad():
        #         output = self._model(input_ids, attention_mask = input_mask, token_type_ids = segment_ids)[0].detach()
        if self._soft_prompt:
            # print(input_ids)
            # input()
            mask = torch.zeros_like(input_ids)
            normal_ids = torch.where(input_ids >= 0, input_ids, mask)
            soft_prompt_ids = -torch.where(input_ids < 0, input_ids, mask)
            normal_embeddings = self.model_embedding(normal_ids)
            soft_prompt_embeddings = self.soft_embedding(soft_prompt_ids)
            soft_prompt_embeddings = self.new_lstm_head(soft_prompt_embeddings)[0]
            soft_prompt_embeddings = self.new_mlp_head(soft_prompt_embeddings)
            input_embeddings = torch.where((input_ids >= 0).unsqueeze(-1), normal_embeddings, soft_prompt_embeddings)
            # print(input_embeddings)
            # input()
            output = self._model(inputs_embeds=input_embeddings)[0]
            # output = self.new_lstm_head(output)[0]

            # print(output)
            # input()
        else:
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
        masked_token_logits = torch.gather(output, 1, masked_token_pos)
        # print(masked_token_logits.shape)
        #masked_token_logits = masked_token_logits.squeeze()  # batch_size * vocab_size
        #rel_and_irrel_logits = masked_token_logits[:, [self._neg_word_id, self._pos_word_id]]

        masked_token_logits=masked_token_logits.reshape(-1,vocab_size)
        rel_and_irrel_logits = masked_token_logits[:, [self._neg_word_id, self._pos_word_id]]
        # print()
        # rel_and_irrel_logits = F.softmax(rel_and_irrel_logits, dim=1)
        # print(rel_and_irrel_logits)

        return rel_and_irrel_logits, masked_token_logits

    def save_prompts(self, file):
        embedding_numpy = np.array(self.soft_embedding.weight.data)
        with open(file, "wb") as f:
            pickle.dump(embedding_numpy, f)