import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        max_len: int = 512
    ) -> None:
        super(PositionalEncoder, self).__init__()
        self._embed_dim = embed_dim
        self._max_len = max_len

        self._embed_matrix = torch.tensor(
            [[pos / pow(1.0e4, 2.0 * (i // 2) / self._embed_dim) for i in range(self._embed_dim)] for pos in range(self._max_len)]
        )
        self._embed_matrix[:, 0::2] = torch.sin(self._embed_matrix[:, 0::2])
        self._embed_matrix[:, 1::2] = torch.cos(self._embed_matrix[:, 1::2])
        self._embedder = nn.Embedding(self._max_len, self._embed_dim)
        self._embedder.weight = nn.Parameter(self._embed_matrix, requires_grad=False)

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        token_len = embed.size()[1]
        if embed.is_cuda:
            ids = torch.cuda.LongTensor([l for l in range(token_len)])
        else:
            ids = torch.LongTensor([l for l in range(token_len)])
        embed += self._embedder(ids)
        return embed
