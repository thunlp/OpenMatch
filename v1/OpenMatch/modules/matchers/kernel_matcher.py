from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class KernelMatcher(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        kernel_num: int = 21
    ) -> None:
        super(KernelMatcher, self).__init__()
        self._embed_dim = embed_dim
        self._kernel_num = kernel_num
        mus, sigmas = self.kernel_init(self._kernel_num)
        self._mus = nn.Parameter(mus, requires_grad=False)
        self._sigmas = nn.Parameter(sigmas, requires_grad=False)

    def kernel_init(self, kernel_num: int) -> Dict[str, torch.Tensor]:
        mus = [1]
        bin_size = 2.0/(kernel_num-1)
        mus.append(1-bin_size/2)
        for i in range(1, kernel_num-1):
            mus.append(mus[i]-bin_size)
        mus = torch.tensor(mus).view(1, 1, 1, kernel_num)

        sigmas = [0.001]
        sigmas += [0.1]*(kernel_num-1)
        sigmas = torch.tensor(sigmas).view(1, 1, 1, kernel_num)
        return mus, sigmas

    def forward(self, k_embed: torch.Tensor, k_mask: torch.Tensor, v_embed: torch.Tensor, v_mask: torch.Tensor) -> torch.Tensor:
        k_embed = k_embed * k_mask.unsqueeze(-1)
        v_embed = v_embed * v_mask.unsqueeze(-1)
        k_by_v_mask = torch.bmm(k_mask.float().unsqueeze(-1), v_mask.float().unsqueeze(-1).transpose(1, 2))
        k_norm = F.normalize(k_embed, p=2, dim=2, eps=1e-10)
        v_norm = F.normalize(v_embed, p=2, dim=2, eps=1e-10)
        inter = (torch.bmm(k_norm, v_norm.transpose(1, 2)) * k_by_v_mask).unsqueeze(-1)

        kernel_outputs = torch.exp((-((inter-self._mus)**2)/(self._sigmas**2)/2))
        kernel_outputs = kernel_outputs.sum(dim=2).clamp(min=1e-10).log() * 1e-2
        logits = kernel_outputs.sum(dim=1)
        return logits
