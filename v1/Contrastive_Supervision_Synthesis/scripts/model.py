import os
import math
import torch
from torch import nn, optim
import logging
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import utils
from contrastqg import (T5ForConditionalGeneration)

logger = logging.getLogger()

class QGenerator(object):
    def __init__(self, args, tokenizer):
        self.network = T5ForConditionalGeneration.from_pretrained(args.pretrain_generator_type)
        self.network.resize_token_embeddings(len(tokenizer))
        self.network.load_state_dict(torch.load(args.generator_load_dir + '/models.pkl'))
        logger.info("sccuess load checkpoint from {} !".format(args.generator_load_dir))
        self.tokenizer = tokenizer
        self.batchify_inputs = utils.select_gen_input_refactor(args)

        
    def predict(self, inputs):        
        self.network.eval()
        outputs = self.network.generate(**inputs)
        pred_tokens = self.tokenizer.convert_outputs_to_tokens(outputs)
        return pred_tokens

    def set_device(self, device):
        self.device = device
        self.network.to(self.device)
        
  
    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)