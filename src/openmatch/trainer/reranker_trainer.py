# Adapted from Tevatron (https://github.com/texttron/tevatron)

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import nested_detach

logger = logging.getLogger(__name__)


class RRTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(RRTrainer, self).__init__(*args, **kwargs)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

    def _prepare_inputs(
            self,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.args.device))
            else:
                prepared.append(super()._prepare_inputs(x))
        return prepared

    def prediction_step(
        self,
        model: nn.Module,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        with torch.no_grad():
            with self.autocast_smart_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()

            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
            else:
                logits = outputs[1:]
        
        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, None)


    def compute_loss(self, model, inputs, return_outputs=False):
        pos_pairs, neg_pairs = inputs
        outputs = model(pos_pairs=pos_pairs, neg_pairs=neg_pairs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

