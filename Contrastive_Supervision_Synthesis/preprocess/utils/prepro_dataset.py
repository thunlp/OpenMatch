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
        '--output_path', 
        type=str, 
        required=True,
        help="Output path of preprocessed dataset."
    )
    
def create_folder_fct(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        
        
def load_jsonl(file_path):
    """ Load file.jsonl ."""
    data_list = []
    with open(file_path, mode='r', encoding='utf-8') as fi:
        for idx, line in enumerate(tqdm(fi)):
            jsonl = json.loads(line)
            data_list.append(jsonl)
    return data_list
        
        
def convert_corpus(data_list):
    data_dict = {}
    for data in tqdm(data_list):
        docid = data["_id"]
        title = data["title"].strip()
        text = data["text"].strip()
        doc = " ".join([title, text]).strip()
        if docid not in data_dict:
            data_dict[docid] = doc
    return data_dict


def save_jsonl(docid2doc, pos_docids, ouput_path):
    
    with open(os.path.join(ouput_path, "docid2doc.jsonl"), 'w', encoding='utf-8') as fd, \
    open(os.path.join(ouput_path, "pos_docids.jsonl"), 'w', encoding='utf-8') as fs:
        for docid in pos_docids:
            fd.write("{}\n".format(json.dumps({"docid":docid, "doc":docid2doc[docid]})))
            fs.write("{}\n".format(json.dumps({"pos_docid":docid})))
        fd.close()
        fs.close()
        
    ## independent folder for bm25 index
    corpus_path = os.path.join(ouput_path, "corpus")
    create_folder_fct(corpus_path)
    
    with open(os.path.join(corpus_path, "corpus.jsonl"), 'w', encoding='utf-8') as fc:
        for docid in docid2doc.keys():
            fc.write("{}\n".format(json.dumps({"id":docid, "contents":docid2doc[docid]})))
        fc.close()

        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        'convert_beir_to_qgformat', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )    
    add_default_args(parser)
    args = parser.parse_args()
    
    args.input_path = os.path.join(args.input_path, args.dataset_name)
    args.output_path = os.path.join(args.output_path, args.dataset_name)

    # create output folder
    create_folder_fct(args.output_path)
    
    ## ******************************
    ## load jsonl
    corpus = load_jsonl(os.path.join(args.input_path, "corpus.jsonl"))
    print("[%s] dataset tot has {%d}"%(args.dataset_name, len(corpus)))
    
    ## ******************************
    ## convert format
    docid2doc = convert_corpus(corpus)
    pos_docids = list(docid2doc.keys())
    random.shuffle(pos_docids)
    
    ## ******************************
    ## convert prepro dataset
    save_jsonl(docid2doc=docid2doc, pos_docids=pos_docids[:500000], ouput_path=args.output_path)
    print("success save {%d} prepro examples"%len(pos_docids))