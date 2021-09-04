import os
import sys
import time
import tqdm
import json
import torch
import random
import logging
import argparse
import traceback
import numpy as np
from tqdm import tqdm

sys.path.append("..")
import utils
import config
from contrastqg import dataloaders
from model import QGenerator
torch.backends.cudnn.benchmark=True


logger = logging.getLogger()


def do_inference(args, generate_loader, generate_dataset, generator):

    torch.cuda.empty_cache()

    gen_qid2query = []
    gen_examples = []
    for step, batch in enumerate(tqdm(generate_loader)):
        inputs, indexs = generator.batchify_inputs(
            **{
            "batch":batch, 
            "device":generator.device, 
            "max_gen_len":args.max_gen_len,
            "top_p":args.top_p, 
            "temperature":args.temperature}
        )
        outputs = generator.predict(inputs)

        for index, output in zip(indexs, outputs):
            ex = generate_dataset.examples[index]
            qid = "%s_%s_%d"%(
                args.generator_mode, 
                args.pretrain_generator_type, 
                len(gen_qid2query)
            )
            if args.generator_mode == "contrastqg":
                gen_examples.append({
                    "qid":qid, 
                    "pos_docid":ex["pos_docid"],
                    "neg_docid":ex["neg_docid"],
                })
            else:
                gen_examples.append({
                    "qid":qid, 
                    "pos_docid":ex["pos_docid"],
                })
                
            gen_qid2query.append({"qid":qid, "query":" ".join(output)})
    
    torch.cuda.empty_cache()
    return gen_qid2query, gen_examples


# -------------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # setting args
    parser = argparse.ArgumentParser(
        'ContrastQG', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    config.add_default_args(parser)
    args = parser.parse_args()
    config.init_args_config(args)
    
    # Setup CUDA, GPU & distributed training    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    logger.info("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # random seed
    utils.set_seed(args)
    
    ## **********************************************
    # load tokenizer
    tokenizer = dataloaders.select_tokenizer(args)
    
    ## **********************************************
    # Load CQG generator
    generator = QGenerator(args, tokenizer=tokenizer["gen_tokenizer"])
    
    # set model device
    generator.set_device(args.device)
    
#     if args.n_gpu > 1:
#         generator.parallelize()
#         logger.info("data parallelize inference ...")
    ## **********************************************
    # Bild generate dataset for CQG
    ## **********************************************
    
    # select dataloader
    dataloder_dict = dataloaders.select_data_loader(args)
    
    generate_dataset = dataloder_dict["build_generate_dataset"](
        args=args, 
        data_dir=args.target_dataset_dir,
        tokenizer=tokenizer["gen_tokenizer"],
    )
    args.gen_batch_size = int(args.per_gpu_gen_batch_size * args.n_gpu)
    logger.info("generation batch size = {}".format(args.gen_batch_size))
    
    gen_sampler = torch.utils.data.sampler.SequentialSampler(generate_dataset)
    gen_data_loader = torch.utils.data.DataLoader(
        generate_dataset,
        batch_size=args.gen_batch_size,
        sampler=gen_sampler,
        num_workers=args.data_workers,
        collate_fn=dataloder_dict["gen_batchify"],
        pin_memory=True,
    )

    ## ***********************
    # [4] Generator Inference
    gen_qid2query, gen_examples = do_inference(**{
        "args":args, 
        "generate_loader":gen_data_loader, 
        "generate_dataset":generate_dataset, 
        "generator":generator})
    
    ## ***********************
    # [5] Save files
    save_folder = os.path.join(args.target_dataset_dir, "%s_%s"%(args.generator_mode, args.pretrain_generator_type))
    utils.create_folder_fct(save_folder)
    
    
    utils.save_list2jsonl(
        data_list=gen_qid2query,
        save_filename=os.path.join(save_folder, "qid2query.jsonl")
    )
    utils.save_list2jsonl(
        data_list=gen_examples, 
        save_filename=os.path.join(save_folder, "examples.jsonl")
    )
    if args.generator_mode == "qg":
        utils.save_list2tsv(
            data_list=gen_qid2query,
            save_filename=os.path.join(save_folder, "qid2query.tsv")
        )