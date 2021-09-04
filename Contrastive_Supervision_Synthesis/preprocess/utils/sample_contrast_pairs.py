import os
import sys
import json
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm


def add_default_args(parser):
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        required=True,
        help="Input path of orignal dataset path."
    )
    parser.add_argument(
        '--input_path', 
        type=str, 
        required=True,
        help="Input path of orignal dataset path."
    )
    parser.add_argument(
        "--generator_folder", 
        choices=["qg_t5-small", "qg_t5-base"],
        required=True,
        type=str,
        help="select generator folder.",
    )
    parser.add_argument(
        '--topk', 
        type=int,
        default=100,
        help="Number of retrieved depth."
    )
    parser.add_argument(
        '--sample_n', 
        type=int,
        default=5,
        help="Number of doc pairs per query."
    )
    

    
def load_trec(input_file, topk):
    """
    Convert base retrieval scores to qid2docids & qid2docid2scores.
    """
    qid2docids = {}
    qid_list = []
    with open(input_file, 'r', encoding='utf-8') as reader:
        for line in tqdm(reader):
            line = line.strip('\n').split(' ')
            assert len(line) == 6
            qid, _, docid, rank, score, _ = line
            
            if int(rank) > topk:
                continue
                
            if qid not in qid_list:
                qid_list.append(qid)
            
            # qid2did_score
            if qid not in qid2docids:
                qid2docids[qid] = set()
                qid2docids[qid].add(docid)
            else:
                qid2docids[qid].add(docid)
    return qid2docids, qid_list

    

def sample_contast_pairs(qid2docids, qid_list, sample_n):
    pair_list = []
    pair_ids = []
    for qid in tqdm(qid_list):
        docids = list(qid2docids[qid])
        random.shuffle(docids)
        sample_num = args.sample_n if args.sample_n <= int(len(docids)/2) else int(len(docids)/2)
        for i in range(0, int(sample_num*2), 2):
            pair_id = "%s-%s"%(docids[i], docids[i+1])
            if pair_id in pair_ids:
                continue
            pair_list.append({"qid":qid, "pos_docid":docids[i], "neg_docid":docids[i+1]})
            pair_ids.append(pair_id)
    return pair_list


def save_list2jsonl(data_list, save_filename):
    with open(file=save_filename, mode="w", encoding="utf-8") as fw:
        for data in data_list:
            fw.write("{}\n".format(json.dumps(data)))
        fw.close()
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'sample_contrast_pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_default_args(parser)
    args = parser.parse_args()
    
    args.input_path = os.path.join(args.input_path, "%s/%s"%(args.dataset_name, args.generator_folder))
    
    # load retrieval files
    bm25_filename = os.path.join(args.input_path, "bm25_retrieval.trec")
    qid2docids, qid_list = load_trec(bm25_filename, topk=args.topk)
    
    # sample pairs
    sample_pairs = sample_contast_pairs(qid2docids, qid_list, sample_n=args.sample_n)
    
    # save pairs
    save_list2jsonl(sample_pairs, os.path.join(args.input_path, "contrast_pairs.jsonl"))