import copy
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import (AutoModel, BatchEncoding, PreTrainedModel,
                          T5EncoderModel, PreTrainedTokenizer, AutoConfig, T5ForConditionalGeneration)
from transformers.modeling_outputs import ModelOutput

from ..arguments import DataArguments
from ..arguments import RRTrainingArguments as TrainingArguments
from ..arguments import ModelArguments
from ..loss import rr_loss_functions, CrossEntropyLoss
from ..utils import mean_pooling
from .linear import LinearHead

logger = logging.getLogger(__name__)


@dataclass
class RROutput(ModelOutput):
    pos_pair_scores: Tensor = None
    neg_pair_scores: Tensor = None
    loss: Tensor = None


class RRModel(nn.Module):
    def __init__(
            self,
            lm: PreTrainedModel,
            head: nn.Module,
            feature: str = "last_hidden_state",
            pooling: str = "first",
            pos_token: str = None,
            neg_token: str = None,
            tokenizer: PreTrainedTokenizer = None,
            model_args: ModelArguments = None,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
    ):
        super().__init__()

        self.lm = lm
        self.head = head

        self.feature = feature
        self.pooling = pooling

        self.pos_token = pos_token
        self.neg_token = neg_token
        self.tokenizer = tokenizer
        self.pos_token_id = tokenizer.encode(self.pos_token, add_special_tokens=False)[0] if self.pos_token else None
        self.neg_token_id = tokenizer.encode(self.neg_token, add_special_tokens=False)[0] if self.neg_token else None

        self.model_args = model_args
        self.data_args = data_args
        self.train_args = train_args

        if train_args is not None:
            self.loss_fn_str = train_args.loss_fn
            self.loss_fn = rr_loss_functions[self.loss_fn_str]()
            self.margin = train_args.margin

        if "T5" in type(self.lm).__name__ and not self.model_args.encoder_only:
            self.loss_fn_str = "ce"
            self.loss_fn = CrossEntropyLoss()

    def _get_config_dict(self):
        config = {
            "plm_backbone": {
                "type": type(self.lm).__name__,
                "feature": self.feature,
            },
            "pooling": self.pooling,
            "pos_token": self.pos_token,
            "neg_token": self.neg_token,
        }
        return config

    def forward(
            self,
            pos_pairs: Dict[str, Tensor] = None,
            neg_pairs: Dict[str, Tensor] = None,
    ):
        pos_pair_scores = self.encode(pos_pairs)
        neg_pair_scores = self.encode(neg_pairs)

        if self.loss_fn_str in ["mr", "smr"]:
            loss = self.loss_fn(pos_pair_scores, neg_pair_scores, margin=self.margin)
        else:
            loss = self.loss_fn(pos_pair_scores, neg_pair_scores)

        return RROutput(
            loss=loss,
            pos_pair_scores=pos_pair_scores,
            neg_pair_scores=neg_pair_scores,
        )

    def encode(self, items):
        if items is None:
            return None, None
        items = BatchEncoding(items)
        if "T5" in type(self.lm).__name__ and not self.model_args.encoder_only:
            decoder_input_ids = torch.zeros((items.input_ids.shape[0], 1), dtype=torch.long).to(items.input_ids.device)
            items_out = self.lm(**items, decoder_input_ids=decoder_input_ids, return_dict=True)
            logits = items_out.logits
            scores = logits[:, 0, [self.neg_token_id, self.pos_token_id]]  # batch_size * 2
        else:
            items_out = self.lm(**items, return_dict=True)
            hidden = getattr(items_out, self.feature)
            if self.pooling == "first":
                reps = hidden[:, 0, :]
            elif self.pooling == "mean":
                reps = mean_pooling(hidden, items.attention_mask)
            else:
                raise ValueError("Unknown pooling type: {}".format(self.pooling))
            scores = self.head(reps)  # batch_size * 1
        return scores

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            data_args: DataArguments = None,
            train_args: TrainingArguments = None,
            tokenizer: PreTrainedTokenizer = None,
            **hf_kwargs,
    ):
        # load local
        config = None
        model_class = None
        hf_config = AutoConfig.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        if model_args.encoder_only:
            model_class = T5EncoderModel
        elif "T5" in hf_config.architectures[0]:  # Pre-trained T5 model
            model_class = T5ForConditionalGeneration
        else:
            model_class = AutoModel
            
        if os.path.exists(os.path.join(model_args.model_name_or_path, "openmatch_config.json")):
            with open(os.path.join(model_args.model_name_or_path, "openmatch_config.json")) as f:
                config = json.load(f)

        if os.path.isdir(model_args.model_name_or_path) and config is not None:  # not a raw Huggingface model
            logger.info(f'loading reranking model weight from {model_args.model_name_or_path}')
            lm = model_class.from_pretrained(
                model_args.model_name_or_path,
                **hf_kwargs
            )
            head = LinearHead.load(ckpt_dir=model_args.model_name_or_path)
        else:  # a Huggingface model
            lm = model_class.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            head = LinearHead(model_args.projection_in_dim, 1)

        model = cls(
            lm=lm,
            head=head,
            feature=model_args.feature if config is None else config["plm_backbone"]["feature"],
            pooling=model_args.pooling if config is None else config["pooling"],
            pos_token=model_args.pos_token if config is None else config["pos_token"],
            neg_token=model_args.neg_token if config is None else config["neg_token"],
            tokenizer=tokenizer,
            model_args=model_args,
            data_args=data_args,
            train_args=train_args,
        )
        return model

    def save(self, output_dir: str):
        self.lm.save_pretrained(output_dir)
        self.head.save(output_dir)

        with open(os.path.join(output_dir, 'openmatch_config.json'), 'w') as f:
            json.dump(self._get_config_dict(), f, indent=4)
