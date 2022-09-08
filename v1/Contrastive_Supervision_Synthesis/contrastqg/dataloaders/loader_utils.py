import os
import re
import json
import logging
from tqdm import tqdm
        
logger = logging.getLogger()




def load_corpus(data_dir):
    """
    :param data_dir: docid2doc
    :param tokenizer: 
    """
    # load docid2doc
    logger.info('start load corpus ...')
    orig_corpus = load_json2dict(
        os.path.join(data_dir, "docid2doc.jsonl"), 
        id_name="docid", 
        text_key="doc",
    )       
    return corpus

    
def load_json2list(file_path):
    """used in load_dataset."""
    data_list = []
    with open(file_path, mode='r', encoding='utf-8') as fi:
        for idx, line in enumerate(tqdm(fi)):
            data = json.loads(line)
            data_list.append(data)
    return data_list


def load_json2dict(file_path, id_name, text_key):
    """used in load_dataset."""
    data_dict = {}
    with open(file_path, mode='r', encoding='utf-8') as fi:
        for idx, line in enumerate(tqdm(fi)):
            data = json.loads(line)
            data_dict[data[id_name]] = data[text_key]
    return data_dict


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def save_tokenized_corpus(dataset, cache_dir):
    """
    :param: dataset dict has keys : docid2doc
    :param: save dir
    """
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    save_dict2jsonl(
        data_dict=dataset["docid2doc"], 
        output_path=os.path.join(cache_dir, "docid2doc.jsonl"), 
        id_name="docid", 
        text_key="doc"
    )
    
