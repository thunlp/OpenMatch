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
from tensorboardX import SummaryWriter

logger = logging.getLogger()



def do_train(args, generator, tot_steps, writer):

    train_loss = utils.AverageMeter()

    for step in tqdm(range(tot_steps)):
        inputs, indexs = generator.generate_train_inputs()
        try:
            loss = generator.update(step, inputs)
        except:
            logging.error(str(traceback.format_exc()))
            break

        train_loss.update(loss)

        # log loss
        if (step + 1) % int(args.display_iter * args.gradient_accumulation_steps) == 0:
            writer.add_scalar('meta_train/loss', train_loss.avg, generator.updates)
            train_loss.reset()

        # save checkpoint
        if (step + 1) % int(args.save_checkpoint_step * args.gradient_accumulation_steps) == 0:
            generator.save_checkpoint(args.checkpoint_folder, generator.updates)



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

    # set tensorboard
    tb_writer = SummaryWriter(args.viso_folder)

    ## **********************************************
    # load tokenizer
    tokenizer = dataloaders.select_tokenizer(args)

    ## **********************************************
    # load CQG generator
    generator = QGenerator(args, tokenizer=tokenizer)

    ## **********************************************
    # initial optimizer
    generator.init_optimizer()

    # clear grad
    generator.zero_grad()

    # set model device
    generator.set_device(args.device)

    if args.n_gpu > 1:
        generator.parallelize()
        logger.info("data parallelize ...")

    ## **********************************************
    # select dataloader
    dataloder_dict = dataloaders.select_data_loader(args)

    train_dataset = dataloder_dict["train_dataset"](
        args=args,
        train_file=args.train_file,
        tokenizer=tokenizer,
    )
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    logger.info("training batch size = {}".format(args.train_batch_size))
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=dataloder_dict["train_batchify"],
        pin_memory=args.cuda,
    )

    generator.reset_train_iter(train_loader=train_data_loader)

    ## **********************************************
    ## train !
    do_train(args, generator, tot_steps=args.max_train_steps, writer=tb_writer)
