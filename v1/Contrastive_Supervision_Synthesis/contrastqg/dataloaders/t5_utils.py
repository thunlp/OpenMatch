import os
import re
import torch
import json
import logging
from tqdm import tqdm
from ..transformers import T5Tokenizer
        
logger = logging.getLogger()

T5_MAX_LEN = 512


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class T5_Tokenizer:
    def __init__(self, args):
        self.args = args
        self.subtokenizer = T5Tokenizer.from_pretrained(self.args.generator_load_dir)

        logger.info("success loaded T5-Tokenizer !")
        
        self.special_tokens = ['<|NEG|>', '<|POS|>']
        [self.neg_token_id, self.pos_token_id] = self.subtokenizer.convert_tokens_to_ids(self.special_tokens)
        self.eos_token_id = self.subtokenizer.eos_token_id
        self.pad_token_id = self.subtokenizer.pad_token_id
    
    def convert_docpair_to_ids(self, pos_doc, neg_doc, max_length):
        pos_doc_ids = self.subtokenizer.encode(pos_doc, max_length=max_length)
        neg_doc_ids = self.subtokenizer.encode(neg_doc, max_length=max_length)
        
        if self.args.reverse_genseq:
            input_ids = [self.neg_token_id] + neg_doc_ids + [self.pos_token_id] + pos_doc_ids + [self.eos_token_id]
        else:
            input_ids = [self.pos_token_id] + pos_doc_ids + [self.neg_token_id] + neg_doc_ids + [self.eos_token_id]
        return input_ids
    
    def convert_doc_to_ids(self, pos_doc, max_length):
        pos_doc_ids = self.subtokenizer.encode(pos_doc, max_length=max_length)
        input_ids = [self.pos_token_id] + pos_doc_ids + [self.eos_token_id]
        return input_ids
        
    
    def convert_outputs_to_tokens(self, outputs):
        batch_text = self.subtokenizer.batch_decode(outputs)
        clean_batch_text = [self.remove_special_tokens(text) for text in batch_text]
        return clean_batch_text
    
    def __len__(self):
        return len(self.subtokenizer)
        
    def remove_special_tokens(self, text):
        # Remove special tokens from the output text in rare cases
        for token in self.special_tokens:
            text = text.replace(token, "")
        return text.split()


## ----------------------------------------------------------------------
## ----------------------------------------------------------------------

def t5_pair_converter(
    index, 
    ex, 
    dataset, 
    tokenizer, 
    max_doc_len=(T5_MAX_LEN - 3) // 2
):
    """
    :param index: training examples list index
    :param ex: one example
    :param dataset: docid2doc
    :param tokenizer: T5 Tokenzier
    :param max_len: max doc tensor length
    :return index, input_ids_tensor, attention_mask_tensor
    """
    
    # pos & neg input text
    pos_doc = dataset["docid2doc"][ex["pos_docid"]]
    neg_doc = dataset["docid2doc"][ex["neg_docid"]]
    
#     pos_doc = " ".join(dataset["docid2doc"][ex["pos_docid"]][:max_doc_len])
#     neg_doc = " ".join(dataset["docid2doc"][ex["neg_docid"]][:max_doc_len])
    
    input_ids = tokenizer.convert_docpair_to_ids(
        pos_doc=pos_doc, 
        neg_doc=neg_doc, 
        max_length=max_doc_len
    )
    return {
        "index":index, 
        "input_tensor":torch.LongTensor(input_ids),
    }


def t5_single_converter(
    index, 
    ex, 
    dataset, 
    tokenizer, 
    max_doc_len=(T5_MAX_LEN - 2)
):
    """
    :param index: training examples list index
    :param ex: one example
    :param dataset: docid2doc
    :param tokenizer: T5 Tokenzier
    :param max_len: max doc tensor length
    :return index, input_ids_tensor, attention_mask_tensor
    """
    
    # pos & neg input text
    
#     ## input token list
#     pos_doc = " ".join(dataset["docid2doc"][ex["pos_docid"]][:max_doc_len])

    ## input token string
    pos_doc = dataset["docid2doc"][ex["pos_docid"]]
    
    input_ids = tokenizer.convert_doc_to_ids(
        pos_doc=pos_doc,
        max_length=max_doc_len
    )
    return {
        "index":index, 
        "input_tensor":torch.LongTensor(input_ids),
    }



def t5_batchify_for_test(batch):
    
    indexs = [ex["index"] for ex in batch]
    inputs = [ex["input_tensor"] for ex in batch]
    
    # pack batch tensor
    batch_tensor = pack_batch_tensor(inputs=inputs)

    return {
        "indexs":indexs,
        "input_ids":batch_tensor["input_ids"], 
        "input_mask":batch_tensor["input_mask"], 
    }

def pack_batch_tensor(inputs):
    """default pad_ids = 0
    """
    input_max_length = max([d.size(0) for d in inputs])
    # prepare batch tensor
    input_ids = torch.LongTensor(len(inputs), input_max_length).zero_()
    input_mask = torch.LongTensor(len(inputs), input_max_length).zero_()
    for i, d in enumerate(inputs):
        input_ids[i, :d.size(0)].copy_(d)
        input_mask[i, :d.size(0)].fill_(1)
    return {
        "input_ids":input_ids, 
        "input_mask":input_mask,
    }

