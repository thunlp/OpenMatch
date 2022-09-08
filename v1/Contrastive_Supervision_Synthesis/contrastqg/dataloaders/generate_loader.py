import os
import logging
import torch
from torch.utils.data import Dataset

from . import loader_utils
from .t5_utils import t5_pair_converter, t5_single_converter
logger = logging.getLogger()



generate_feature_converter = {
    "contrastqg":t5_pair_converter,
    "qg":t5_single_converter
}


class generate_dataset(Dataset):
    def __init__(
        self, 
        args,
        data_dir,
        tokenizer, 
    ):
        """
        :param intput_dir: examples.jsonl ("pos_docid"/"neg_docid"); docid2doc.jsonl
        :param tokenizer: T5Tokenizer or None
        """
        # load pairs {"pos_docid", "neg_docid"}
        if args.generator_mode == "contrastqg":
            examples = loader_utils.load_json2list(os.path.join(data_dir, "qg_%s/contrast_pairs.jsonl"%args.pretrain_generator_type))
        else:
            examples = loader_utils.load_json2list(os.path.join(data_dir, "pos_docids.jsonl"))
        logger.info('[%s] needs generate %d examples'%(args.generator_mode, len(examples)))
        
        # load docid2doc {"docid":doc}
        docid2doc = loader_utils.load_json2dict(
            os.path.join(data_dir, "docid2doc.jsonl"), 
            id_name="docid", 
            text_key="doc",
        )
        
        # load docid2doc
        self.args = args
        self.dataset = {"docid2doc":None, "qid2query":None}
        self.dataset["docid2doc"] = docid2doc
        self.tokenizer = tokenizer
        self.examples = examples
                
    def __len__(self):
        return len(self.examples)
    
    def reset_examples(self, examples):
        self.examples = examples
    
    def reset_qid2query(self, qid2query):
        self.dataset["qid2query"] = qid2query

    def __getitem__(self, index):
        return generate_feature_converter[self.args.generator_mode](
            index,
            ex=self.examples[index],
            dataset=self.dataset, 
            tokenizer=self.tokenizer,
        )