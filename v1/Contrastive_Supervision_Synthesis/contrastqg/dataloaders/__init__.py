def select_tokenizer(args): 
    if "t5" in args.pretrain_generator_type:
        return {"gen_tokenizer":T5_Tokenizer(args)}
    raise ValueError('Invalid generator class: %s' % args.pretrain_generator_type)
    
    
def select_data_loader(args, do_finetune=False):
    dataloder_dict = {"build_generate_dataset":generate_dataset}
    
    if "t5" in args.pretrain_generator_type:
        dataloder_dict["gen_batchify"] = t5_batchify_for_test
        return dataloder_dict
    
    raise ValueError('Invalid generator class: %s' % args.pretrain_generator_type)
    

from .generate_loader import generate_dataset
from .t5_utils import (
    T5_Tokenizer,
    t5_batchify_for_test,
)
