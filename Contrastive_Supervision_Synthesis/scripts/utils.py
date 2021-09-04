import os
import csv
import time
import json
import torch
import random
import argparse
import logging
import traceback
import numpy as np
from typing import List, Dict

import pytrec_eval

logger = logging.getLogger()


# ------------------------------------------------------------
# ------------------------------------------------------------
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
## ------------------------------------------------------------
## ------------------------------------------------------------
## T5 Refactor
def select_gen_input_refactor(args):
    if "t5" in args.pretrain_generator_type:
        return t5_refactor_for_test
    raise ValueError('Invalid generator class: %s' % args.pretrain_generator_type)
    
    
def t5_refactor_for_test(batch, device, max_gen_len, top_p, temperature):
    inputs = {
        "input_ids":batch["input_ids"].to(device), 
        "attention_mask":batch["input_mask"].to(device), 
        "max_length":max_gen_len,
        "top_p":top_p,
        "temperature":temperature,
        
    }
    return inputs, batch["indexs"]


## ------------------------------------------------------------
## ------------------------------------------------------------
## Save files

def create_folder_fct(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        
def save_list2jsonl(data_list, save_filename):
    with open(file=save_filename, mode="w", encoding="utf-8") as fw:
        for data in data_list:
            fw.write("{}\n".format(json.dumps(data)))
        fw.close()
        
def save_dict2jsonl(data_dict, save_filename, id_name, text_key):
    with open(file=save_filename, mode="w", encoding="utf-8") as fw:
        for key in data_dict:
            data = {id_name:key, text_key:data_dict[key]}
            fw.write("{}\n".format(json.dumps(data)))
        fw.close()
        

def save_list2tsv(data_list, save_filename):
    with open(file=save_filename, mode="w", encoding="utf-8") as fw:
        tsv_w = csv.writer(fw, delimiter='\t')
        for data in data_list:
            tsv_w.writerow([data["qid"], data["query"]])
        fw.close()
        