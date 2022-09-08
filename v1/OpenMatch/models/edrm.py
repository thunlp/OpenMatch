from typing import List, Tuple

import torch
import torch.nn as nn

from OpenMatch.modules.embedders import Embedder
from OpenMatch.modules.encoders import Conv1DEncoder
from OpenMatch.modules.matchers import KernelMatcher

class EDRM(nn.Module):
    def __init__(
        self,
        wrd_vocab_size: int,
        ent_vocab_size: int,
        wrd_embed_dim: int,
        ent_embed_dim: int,
        max_des_len: int = 20,
        max_ent_num: int = 3,
        kernel_num: int = 21,
        kernel_dim: int = 128,
        kernel_sizes: List[int] = [1, 2, 3],
        wrd_embed_matrix: List[float] = None,
        ent_embed_matrix: List[float] = None,
        task: str = 'ranking'
    ) -> None:
        super(EDRM, self).__init__()
        self._wrd_vocab_size = wrd_vocab_size
        self._ent_vocab_size = ent_vocab_size
        self._wrd_embed_dim = wrd_embed_dim
        self._ent_embed_dim = ent_embed_dim
        self._max_des_len = max_des_len
        self._max_ent_num = max_ent_num
        self._kernel_num = kernel_num
        self._kernel_dim = kernel_dim
        self._kernel_sizes = kernel_sizes
        self._wrd_embed_matrix = wrd_embed_matrix
        self._ent_embed_matrix = ent_embed_matrix
        self._task = task
        if self._ent_embed_dim != self._kernel_dim:
            raise ValueError('ent_embed_dim must equal to kernel_dim.')

        self._wrd_embedder = Embedder(self._wrd_vocab_size, self._wrd_embed_dim, self._wrd_embed_matrix)
        self._ent_embedder = Embedder(self._ent_vocab_size, self._ent_embed_dim, self._ent_embed_matrix)
        self._wrd_encoder = Conv1DEncoder(self._wrd_embed_dim, self._kernel_dim, self._kernel_sizes)
        self._des_encoder = Conv1DEncoder(self._wrd_embed_dim * self._max_ent_num, self._kernel_dim, [1])
        self._des_maxpool = nn.MaxPool1d(self._max_des_len - self._max_ent_num + 1)
        self._matcher = KernelMatcher(self._wrd_encoder.get_output_dim(), self._kernel_num)
        if self._task == 'ranking':
            self._dense = nn.Linear(self._kernel_num * ((len(self._kernel_sizes) + 1) ** 2), 1)
        elif self._task == 'classification':
            self._dense = nn.Linear(self._kernel_num * ((len(self._kernel_sizes) + 1) ** 2), 2)
        else:
            raise ValueError('Task must be `ranking` or `classification`.')

    def forward(self, query_wrd_ids: torch.Tensor, query_wrd_masks: torch.Tensor, doc_wrd_ids: torch.Tensor, doc_wrd_masks: torch.Tensor, query_ent_ids: torch.Tensor, query_ent_masks: torch.Tensor, doc_ent_ids: torch.Tensor, doc_ent_masks: torch.Tensor, query_des_ids: torch.Tensor, doc_des_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query_wrd_embed = self._wrd_embedder(query_wrd_ids)
        doc_wrd_embed = self._wrd_embedder(doc_wrd_ids)
        query_ent_embed = self._ent_embedder(query_ent_ids)
        doc_ent_embed = self._ent_embedder(doc_ent_ids)
        query_des_embed = self._wrd_embedder(query_des_ids)
        doc_des_embed = self._wrd_embedder(doc_des_ids)

        _, query_encs = self._wrd_encoder(query_wrd_embed, query_wrd_masks)
        _, doc_encs = self._wrd_encoder(doc_wrd_embed, doc_wrd_masks)

        batch_size = query_wrd_embed.size()[0]
        _, query_des_encs = self._des_encoder(query_des_embed.view(batch_size, -1, self._wrd_embed_dim * self._max_ent_num))
        _, doc_des_encs = self._des_encoder(doc_des_embed.view(batch_size, -1, self._wrd_embed_dim * self._max_ent_num))
        query_encs.append(query_ent_embed + self._des_maxpool(query_des_encs[0].transpose(1, 2)).transpose(1, 2))
        doc_encs.append(doc_ent_embed + self._des_maxpool(doc_des_encs[0].transpose(1, 2)).transpose(1, 2))

        logits = torch.cat([self._matcher(query_enc, query_wrd_masks[:, :query_enc.size()[1]] if i+1 != len(query_encs) else query_ent_masks[:, :query_enc.size()[1]], doc_enc, doc_wrd_masks[:, :doc_enc.size()[1]] if j+1 != len(doc_encs) else doc_ent_masks[:, :doc_enc.size()[1]]) for i, query_enc in enumerate(query_encs) for j, doc_enc in enumerate(doc_encs)], dim=1)
        score = self._dense(logits).squeeze(-1)
        return score, logits
