import logging
import os
import json

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class LinearHead(nn.Module):
    def __init__(
            self,
            input_dim: int = 768,
            output_dim: int = 768,
    ):
        super(LinearHead, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.config = {'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, rep: Tensor = None):
        return self.linear(rep)

    @classmethod
    def load(cls, ckpt_dir: str):
        logger.info(f'Loading linear head from {ckpt_dir}')
        model_path = os.path.join(ckpt_dir, 'linear.pt')
        config_path = os.path.join(ckpt_dir, 'head_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        model = cls(**config)
        model.load_state_dict(torch.load(model_path))
        return model

    def save(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'linear.pt'))
        with open(os.path.join(save_path, 'head_config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)