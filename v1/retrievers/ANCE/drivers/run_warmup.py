import sys
sys.path += ["../"]
import pandas as pd
from transformers import glue_compute_metrics as compute_metrics, glue_output_modes as output_modes, glue_processors as processors
from transformers import (
    AdamW,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    RobertaModel,
)
import transformers
from utils.eval_mrr import passage_dist_eval
from model.models import MSMarcoConfigDict
from utils.lamb import Lamb
import os
from os import listdir
from os.path import isfile, join
import argparse
import glob
import json
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from utils.util import getattr_recursive, set_seed, is_first_worker, StreamingDataset
from utils.util import (
    StreamingDataset, 
    EmbeddingCache, 
    get_checkpoint_no, 
    get_latest_ann_data,
    barrier_array_merge,
    is_first_worker,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def train(args, model, tokenizer, f, train_fn):
    """ Train the model """
    tb_writer = None
    if is_first_worker():
        tb_writer = SummaryWriter(log_dir=args.log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    real_batch_size = args.train_batch_size * args.gradient_accumulation_steps * \
        (torch.distributed.get_world_size() if args.local_rank != -1 else 1)

    if args.max_steps > 0:
        t_total = args.max_steps
    else:
        t_total = args.expected_train_size // real_batch_size * args.num_train_epochs

    # layerwise optimization for lamb
    optimizer_grouped_parameters = []
    layer_optim_params = set()
    for layer_name in ["roberta.embeddings", "score_out", "downsample1", "downsample2", "downsample3", "embeddingHead"]:
        layer = getattr_recursive(model, layer_name)
        if layer is not None:
            optimizer_grouped_parameters.append({"params": layer.parameters()})
            for p in layer.parameters():
                layer_optim_params.add(p)
    if getattr_recursive(model, "roberta.encoder.layer") is not None:
        for layer in model.roberta.encoder.layer:
            optimizer_grouped_parameters.append({"params": layer.parameters()})
            for p in layer.parameters():
                layer_optim_params.add(p)
    optimizer_grouped_parameters.append(
        {"params": [p for p in model.parameters() if p not in layer_optim_params]})

    if args.optimizer.lower() == "lamb":
        optimizer = Lamb(optimizer_grouped_parameters,
                         lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer.lower() == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        raise Exception(
            "optimizer {0} not recognized! Can only be lamb or adamW".format(args.optimizer))

    if args.scheduler.lower() == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    elif args.scheduler.lower() == "cosine":
        scheduler = CosineAnnealingLR(optimizer, t_total, 1e-8)
    else:
        raise Exception(
            "Scheduler {0} not recognized! Can only be linear or cosine".format(args.scheduler))

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ) and args.load_optimizer_scheduler:
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(
            os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(
            os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[
                args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )
        logger.info("assign the model as torch.nn.parallel.DistributedDataParallel()")

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(
                args.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (args.expected_train_size //
                                             args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (
                args.expected_train_size // args.gradient_accumulation_steps)

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info(
                "  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch",
                        steps_trained_in_current_epoch)
        except:
            logger.info("  Start training from a pretrained model")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for m_epoch in train_iterator:
        f.seek(0)
        sds = StreamingDataset(f,train_fn)
        epoch_iterator = DataLoader(sds, batch_size=args.per_gpu_train_batch_size, num_workers=0)
        for step, batch in tqdm(enumerate(epoch_iterator),desc="Iteration",disable=args.local_rank not in [-1,0]):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device).long() for t in batch)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                outputs = model(*batch)
            else:
                with model.no_sync():
                    outputs = model(*batch)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    loss.backward()
                else:
                    with model.no_sync():
                        loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if is_first_worker() and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(
                        output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(
                        output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(
                        output_dir, "scheduler.pt"))
                    logger.info(
                        "Saving optimizer and scheduler states to %s", output_dir)
                #dist.barrier()

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.evaluate_during_training and global_step % (args.logging_steps_per_eval*args.logging_steps) == 0:
                        model.eval()
                        reranking_mrr, full_ranking_mrr = passage_dist_eval(
                            args, model, tokenizer)
                        if is_first_worker():
                            print(
                                "Reranking/Full ranking mrr: {0}/{1}".format(str(reranking_mrr), str(full_ranking_mrr)))
                            mrr_dict = {"reranking": float(
                                reranking_mrr), "full_raking": float(full_ranking_mrr)}
                            tb_writer.add_scalars("mrr", mrr_dict, global_step)
                            print(args.output_dir)

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    if is_first_worker():
                        for key, value in logs.items():
                            print(key, type(value))
                            tb_writer.add_scalar(key, value, global_step)
                        tb_writer.add_scalar("epoch", m_epoch, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))
                    #dist.barrier()

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        tb_writer.close()

    return global_step, tr_loss / global_step


def load_stuff(model_type, args):
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    args.output_mode = "classification"
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.num_labels = num_labels
    if args.resume_train:
        args.model_name_or_path, _ =  get_latest_checkpoint(args)
    # Load pretrained model and tokenizer
    #if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
    #    torch.distributed.barrier()

    configObj = MSMarcoConfigDict[model_type]
    model_args = type('', (), {})()
    model_args.use_mean = configObj.use_mean
    config = configObj.config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = configObj.tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = configObj.model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
        model_argobj=model_args,
    )

    #if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        #torch.distributed.barrier()

    model.to(args.device)

    return config, tokenizer, model, configObj

def get_latest_checkpoint(args):
    if not os.path.exists(args.output_dir):
        raise ValueError(
            "Output checkpoint directory ({}) not exists. Please remove the --resume_train option".format(
                args.output_dir
            )
        )
    files = list(next(os.walk(args.output_dir))[1])

    def valid_checkpoint(checkpoint):
        return checkpoint.startswith("checkpoint-")

    logger.info("checkpoint files")
    logger.info(files)
    checkpoint_nums = [get_checkpoint_no(s) for s in files if valid_checkpoint(s)]

    if len(checkpoint_nums) > 0:
        return os.path.join(args.output_dir, "checkpoint-" + str(max(checkpoint_nums))), max(checkpoint_nums)
    else:
        raise ValueError(
            "Output checkpoint directory ({}) may have some unepected error.".format(
                args.output_dir
            )
        )

# def load_model(args, checkpoint_path):
#     label_list = ["0", "1"]
#     num_labels = len(label_list)
#     args.model_type = args.model_type.lower()
#     configObj = MSMarcoConfigDict[args.model_type]
#     args.model_name_or_path = checkpoint_path

#     model = configObj.model_class(args)

#     saved_state = load_states_from_checkpoint(checkpoint_path)
#     model_to_load = get_model_obj(model)
#     logger.info('Loading saved model state ...')
#     model_to_load.load_state_dict(saved_state.model_dict)

#     model.to(args.device)
#     logger.info("Inference parameters %s", args)
#     if args.local_rank != -1:
#         model = torch.nn.parallel.DistributedDataParallel(
#             model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
#         )
#     return model

def get_arguments():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--train_model_type",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", 
        default="", 
        type=str, 
        help="Pretrained config name or path if not the same as model_name",
    )

    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--do_train", 
        action="store_true",
        help="Whether to run training.",
    )

    parser.add_argument(
        "--resume_train", 
        action="store_true",
        help="Whether to run training.",
    )

    parser.add_argument(
        "--do_eval", 
        action="store_true",
        help="Whether to run eval on the dev set.",
    )

    parser.add_argument(
        "--evaluate_during_training", 
        action="store_true", 
        help="Rul evaluation during training at each logging step.",
    )

    parser.add_argument(
        "--do_lower_case", 
        action="store_true", 
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="Tensorboard log dir",
    )

    parser.add_argument(
        "--eval_type",
        default="full",
        type=str,
        help="MSMarco eval type - dev full or small",
    )

    parser.add_argument(
        "--optimizer",
        default="lamb",
        type=str,
        help="Optimizer - lamb or adamW",
    )

    parser.add_argument(
        "--scheduler",
        default="linear",
        type=str,
        help="Scheduler - linear or cosine",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", 
        default=8, 
        type=int, 
        help="Batch size per GPU/CPU for training.",
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size", 
        default=8, 
        type=int, 
        help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--learning_rate", 
        default=5e-5,
        type=float, 
        help="The initial learning rate for Adam.",
    )

    parser.add_argument(
        "--weight_decay", 
        default=0.0,
        type=float, 
        help="Weight decay if we apply some.",
    )

    parser.add_argument(
        "--adam_epsilon", 
        default=1e-8,
        type=float, 
        help="Epsilon for Adam optimizer.",
    )

    parser.add_argument(
        "--max_grad_norm", 
        default=1.0,
        type=float, 
        help="Max gradient norm.",
    )

    parser.add_argument(
        "--num_train_epochs", 
        default=3.0, 
        type=float, 
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )

    parser.add_argument(
        "--warmup_steps",
         default=0, 
         type=int,
         help="Linear warmup over warmup_steps.",
    )

    parser.add_argument(
        "--logging_steps", 
        type=int,
        default=500, 
        help="Log every X updates steps.",
    )

    parser.add_argument(
        "--logging_steps_per_eval", 
        type=int,
        default=10, 
        help="Eval every X logging steps.",
    )

    parser.add_argument(
        "--save_steps", 
        type=int, 
        default=500,
        help="Save checkpoint every X updates steps.",
    )

    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )

    parser.add_argument(
        "--no_cuda", 
        action="store_true",
        help="Avoid using CUDA when available",
    )

    parser.add_argument(
        "--overwrite_output_dir", 
        action="store_true", 
        help="Overwrite the content of the output directory",
    )

    parser.add_argument(
        "--overwrite_cache", 
        action="store_true", 
        help="Overwrite the cached training and evaluation sets",
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="random seed for initialization",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument(
        "--expected_train_size",
        default=100000,
        type=int,
        help="Expected train dataset size",
    )

    parser.add_argument(
        "--load_optimizer_scheduler",
        default=False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1,
        help="For distributed training: local_rank",
    )

    parser.add_argument(
        "--server_ip", 
        type=str, 
        default="",
        help="For distant debugging.",
    )

    parser.add_argument(
        "--server_port", 
        type=str,
        default="", 
        help="For distant debugging.",
    )

    args = parser.parse_args()

    return args


def set_env(args):
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
        and not args.resume_train
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)


def save_checkpoint(args, model, tokenizer):
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and is_first_worker():
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    #dist.barrier()


def evaluation(args, model, tokenizer):
    # Evaluation
    results = {}
    if args.do_eval:
        model_dir = args.model_name_or_path if args.model_name_or_path else args.output_dir

        checkpoints = [model_dir]

        for checkpoint in checkpoints:
            global_step = checkpoint.split(
                "-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split(
                "/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model.eval()
            reranking_mrr, full_ranking_mrr = passage_dist_eval(
                args, model, tokenizer)
            if is_first_worker():
                print(
                    "Reranking/Full ranking mrr: {0}/{1}".format(str(reranking_mrr), str(full_ranking_mrr)))
            #dist.barrier()
    return results


def main():
    args = get_arguments()
    set_env(args)

    config, tokenizer, model, configObj = load_stuff(
        args.train_model_type, args)

    # Training
    if args.do_train:
        logger.info("Training/evaluation parameters %s", args)

        def train_fn(line, i):
            return configObj.process_fn(line, i, tokenizer, args)

        with open(args.data_dir+"/triples.train.small.tsv", encoding="utf-8-sig") as f:
            train_batch_size = args.per_gpu_train_batch_size * \
                max(1, args.n_gpu)
            global_step, tr_loss = train(
                args, model, tokenizer, f, train_fn)
            logger.info(" global_step = %s, average loss = %s",
                        global_step, tr_loss)

    save_checkpoint(args, model, tokenizer)

    results = evaluation(args, model, tokenizer)
    return results


if __name__ == "__main__":
    main()
