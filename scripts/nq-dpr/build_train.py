# Adapted from Tevatron (https://github.com/texttron/tevatron)

import json
import os
from argparse import ArgumentParser
from multiprocessing import Pool

from openmatch.utils import fill_template
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--query_template', type=str, default="<question>")
parser.add_argument('--doc_template', type=str, default="<title> [SEP] <text>")
parser.add_argument('--query_max_len', type=int, default=32)
parser.add_argument('--doc_max_len', type=int, default=256)
parser.add_argument('--tokenizer', type=str, required=False, default='bert-base-uncased')
parser.add_argument('--minimum_negatives', type=int, required=False, default=1)
parser.add_argument('--mp_chunk_size', type=int, default=500)
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

data = json.load(open(args.input))

save_dir = os.path.split(args.output)[0]
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

doc_markers = ["title", "text"]


def process_item(item):
    if len(item['hard_negative_ctxs']) < args.minimum_negatives or len(item['positive_ctxs']) < 1:
        return
    group = {}
    positives = []
    for pos in item['positive_ctxs']:
        positives.append(fill_template(args.doc_template, pos, doc_markers))
    negatives = []
    for neg in item['hard_negative_ctxs']:
        negatives.append(fill_template(args.doc_template, neg, doc_markers))

    query = tokenizer.encode(fill_template(args.query_template, item, ["question"]), add_special_tokens=False, max_length=args.query_max_len, truncation=True)
    positives = tokenizer(
            positives, add_special_tokens=False, max_length=args.doc_max_len, truncation=True, padding=False)['input_ids']
    negatives = tokenizer(
            negatives, add_special_tokens=False, max_length=args.doc_max_len, truncation=True, padding=False)['input_ids']

    group['query'] = query
    group['positives'] = positives
    group['negatives'] = negatives

    return group


with Pool(args.num_workers) as p:
    groups = list(tqdm(p.imap(process_item, data, chunksize=args.mp_chunk_size), total=len(data)))

with open(args.output, 'w') as f:
    for group in tqdm(groups):
        if group is None:
            continue
        f.write(json.dumps(group) + "\n")