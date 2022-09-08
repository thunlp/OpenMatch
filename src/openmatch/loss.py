import torch
from torch import Tensor
from torch.nn import functional as F
from torch import distributed as dist


class SimpleContrastiveLoss:

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean'):
        if target is None:
            target_per_qry = y.size(0) // x.size(0)
            target = torch.arange(
                0, x.size(0) * target_per_qry, target_per_qry, device=x.device, dtype=torch.long)
        logits = torch.matmul(x, y.transpose(0, 1))
        return F.cross_entropy(logits, target, reduction=reduction)


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.word_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)


class MarginRankingLoss:
    def __init__(self, margin: float = 1.0):
        self.margin = margin
    
    def __call__(self, pos_scores: Tensor, neg_scores: Tensor):
        return torch.mean(F.relu(self.margin - pos_scores + neg_scores))


class SoftMarginRankingLoss:
    def __init__(self, margin: float = 1.0):
        self.margin = margin
    
    def __call__(self, pos_scores: Tensor, neg_scores: Tensor):
        return torch.mean(F.softplus(self.margin - pos_scores + neg_scores))


class BinaryCrossEntropyLoss:
    def __call__(self, pos_scores: Tensor, neg_scores: Tensor):
        return (F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores)) 
              + F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores)))


class CrossEntropyLoss:
    def __call__(self, pos_scores: Tensor, neg_scores: Tensor):
        return (F.cross_entropy(pos_scores, torch.ones(pos_scores.shape[0], dtype=torch.long).to(pos_scores.device)) 
              + F.cross_entropy(neg_scores, torch.zeros(neg_scores.shape[0], dtype=torch.long).to(pos_scores.device)))


rr_loss_functions = {
    "mr": MarginRankingLoss,
    "smr": SoftMarginRankingLoss,
    "bce": BinaryCrossEntropyLoss,
    "ce": CrossEntropyLoss,
}