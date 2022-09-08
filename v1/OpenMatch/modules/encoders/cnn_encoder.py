from typing import List, Tuple

import torch
import torch.nn as nn

class Conv1DEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        kernel_dim: int,
        kernel_sizes: List[int] = [2, 3, 4, 5],
        stride: int = 1
    ) -> None:
        super(Conv1DEncoder, self).__init__()
        self._embed_dim = embed_dim
        self._kernel_dim = kernel_dim
        self._kernel_sizes = kernel_sizes
        self._stride = stride
        self._output_dim = self._kernel_dim * len(self._kernel_sizes)

        self._encoder = nn.ModuleList([
            nn.Conv1d(
                in_channels=self._embed_dim,
                out_channels=self._kernel_dim,
                kernel_size=kernel_size,
                stride = self._stride
            )
            for kernel_size in self._kernel_sizes
        ])
        self._activation = nn.ReLU()

    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, embed: torch.Tensor, masks: torch.Tensor = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if masks is not None:
            embed = embed * masks.unsqueeze(-1)
        embed = torch.transpose(embed, 1, 2)

        kernel_outputs = [self._activation(enc(embed)) for enc in self._encoder]
        pooling_sums = [kernel_output.max(dim=2).values for kernel_output in kernel_outputs]
        enc = (torch.cat(pooling_sums, dim=1) if len(pooling_sums) > 1 else pooling_sums[0])
        return enc, [torch.transpose(kernel_output, 1, 2) for kernel_output in kernel_outputs]
