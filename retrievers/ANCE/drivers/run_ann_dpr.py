from os.path import join
import sys
sys.path += ['../']
import argparse
import glob
import json
import logging
import os
import random
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from model.models import MSMarcoConfigDict, ALL_MODELS
from utils.lamb import Lamb
import random 
import transformers
from transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_processors as processors
import copy
from torch import nn
import pickle
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
import pandas as pd
logger = logging.getLogger(__name__)
from utils.util import (
    StreamingDataset, 
    EmbeddingCache, 
    get_checkpoint_no, 
    get_latest_ann_data,
    set_seed,
    is_first_worker,
)
from data.DPR_data import GetTrainingDataProcessingFn, GetTripletTrainingDataProcessingFn
from utils.dpr_utils import (
    load_states_from_checkpoint, 
    get_model_obj, 
    CheckpointState, 
    get_optimizer, 
    all_gather_list
)


def train(args, model, tokenizer, query_cache, passage_cache):
    """ Train the model """
    logger.info("Training/evaluation parameters %s", args)
    tb_writer = None
    if is_first_worker():
        tb_writer = SummaryWriter(log_dir=args.log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) #nll loss for query
    real_batch_size = args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    
    optimizer = get_optimizer(args, model, weight_decay=args.weight_decay,)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Max steps = %d", args.max_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    tr_loss = 0.0
    model.zero_grad()
    model.train()
    set_seed(args)  # Added here for reproductibility

    last_ann_no = -1
    train_dataloader = None
    train_dataloader_iter = None
    dev_ndcg = 0
    step = 0
    iter_count = 0

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps= args.max_steps
    )

    global_step = 0
    if args.model_name_or_path != "bert-base-uncased":
        saved_state = load_states_from_checkpoint(args.model_name_or_path)
        global_step = _load_saved_state(model, optimizer, scheduler, saved_state)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from global step %d", global_step)


        nq_dev_nll_loss, nq_correct_ratio = evaluate_dev(args, model, passage_cache)
        dev_nll_loss_trivia, correct_ratio_trivia = evaluate_dev(args, model, passage_cache, "-trivia")
        if is_first_worker():
            tb_writer.add_scalar("dev_nll_loss/dev_nll_loss", nq_dev_nll_loss, global_step)
            tb_writer.add_scalar("dev_nll_loss/correct_ratio", nq_correct_ratio, global_step)
            tb_writer.add_scalar("dev_nll_loss/dev_nll_loss_trivia", dev_nll_loss_trivia, global_step)
            tb_writer.add_scalar("dev_nll_loss/correct_ratio_trivia", correct_ratio_trivia, global_step)

    while global_step < args.max_steps:

        if step % args.gradient_accumulation_steps == 0 and global_step % args.logging_steps == 0:

            if args.num_epoch == 0:
                # check if new ann training data is availabe
                ann_no, ann_path, ndcg_json = get_latest_ann_data(args.ann_dir)
                if ann_path is not None and ann_no != last_ann_no:
                    logger.info("Training on new add data at %s", ann_path)
                    with open(ann_path, 'r') as f:
                        ann_training_data = f.readlines()
                    logger.info("Training data line count: %d", len(ann_training_data))
                    ann_training_data = [l for l in ann_training_data if len(l.split('\t')[2].split(',')) > 1]
                    logger.info("Filtered training data line count: %d", len(ann_training_data))
                    ann_checkpoint_path = ndcg_json['checkpoint']
                    ann_checkpoint_no = get_checkpoint_no(ann_checkpoint_path)

                    aligned_size = (len(ann_training_data) // args.world_size) * args.world_size
                    ann_training_data = ann_training_data[:aligned_size]

                    logger.info("Total ann queries: %d", len(ann_training_data))
                    if args.triplet:
                        train_dataset = StreamingDataset(ann_training_data, GetTripletTrainingDataProcessingFn(args, query_cache, passage_cache))
                        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
                    else:
                        train_dataset = StreamingDataset(ann_training_data, GetTrainingDataProcessingFn(args, query_cache, passage_cache))
                        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size*2)
                    train_dataloader_iter = iter(train_dataloader)

                    # re-warmup
                    if not args.single_warmup:
                        scheduler = get_linear_schedule_with_warmup(
                            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps= len(ann_training_data)
                        )

                    if args.local_rank != -1:
                        dist.barrier()
            
                    if is_first_worker():
                        # add ndcg at checkpoint step used instead of current step
                        tb_writer.add_scalar("retrieval_accuracy/top20_nq", ndcg_json['top20'], ann_checkpoint_no)
                        tb_writer.add_scalar("retrieval_accuracy/top100_nq", ndcg_json['top100'], ann_checkpoint_no)
                        if 'top20_trivia' in ndcg_json:
                            tb_writer.add_scalar("retrieval_accuracy/top20_trivia", ndcg_json['top20_trivia'], ann_checkpoint_no)
                            tb_writer.add_scalar("retrieval_accuracy/top100_trivia", ndcg_json['top100_trivia'], ann_checkpoint_no)
                        if last_ann_no != -1:
                            tb_writer.add_scalar("epoch", last_ann_no, global_step-1)
                        tb_writer.add_scalar("epoch", ann_no, global_step)
                    last_ann_no = ann_no
            elif step == 0:
                train_data_path = os.path.join(args.data_dir, "train-data")
                with open(train_data_path, 'r') as f:
                    training_data = f.readlines()
                if args.triplet:
                    train_dataset = StreamingDataset(training_data, GetTripletTrainingDataProcessingFn(args, query_cache, passage_cache))
                    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
                else:
                    train_dataset = StreamingDataset(training_data, GetTrainingDataProcessingFn(args, query_cache, passage_cache))
                    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size*2)
                all_batch = [b for b in train_dataloader]
                logger.info("Total batch count: %d", len(all_batch))
                train_dataloader_iter = iter(train_dataloader)

        try:
            batch = next(train_dataloader_iter)
        except StopIteration:
            logger.info("Finished iterating current dataset, begin reiterate")
            if args.num_epoch != 0:
                iter_count += 1
                if is_first_worker():
                    tb_writer.add_scalar("epoch", iter_count-1, global_step-1)
                    tb_writer.add_scalar("epoch", iter_count, global_step)
            nq_dev_nll_loss, nq_correct_ratio = evaluate_dev(args, model, passage_cache)
            dev_nll_loss_trivia, correct_ratio_trivia = evaluate_dev(args, model, passage_cache, "-trivia")
            if is_first_worker():
                tb_writer.add_scalar("dev_nll_loss/dev_nll_loss", nq_dev_nll_loss, global_step)
                tb_writer.add_scalar("dev_nll_loss/correct_ratio", nq_correct_ratio, global_step)
                tb_writer.add_scalar("dev_nll_loss/dev_nll_loss_trivia", dev_nll_loss_trivia, global_step)
                tb_writer.add_scalar("dev_nll_loss/correct_ratio_trivia", correct_ratio_trivia, global_step)
            train_dataloader_iter = iter(train_dataloader)
            batch = next(train_dataloader_iter)
            dist.barrier()

        if args.num_epoch != 0 and iter_count > args.num_epoch:
            break
        
        step += 1
        if args.triplet:
            loss = triplet_fwd_pass(args, model, batch)
        else:
            loss, correct_cnt = do_biencoder_fwd_pass(args, model, batch)

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            if step % args.gradient_accumulation_steps == 0:
                loss.backward()
            else:
                with model.no_sync():
                    loss.backward()          

        tr_loss += loss.item()
        if step % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logs = {}
                loss_scalar = tr_loss / args.logging_steps
                learning_rate_scalar = scheduler.get_lr()[0]
                logs["learning_rate"] = learning_rate_scalar
                logs["loss"] = loss_scalar
                tr_loss = 0

                if is_first_worker():
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    logger.info(json.dumps({**logs, **{"step": global_step}}))

            if is_first_worker() and args.save_steps > 0 and global_step % args.save_steps == 0:
                _save_checkpoint(args, model, optimizer, scheduler, global_step)

    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        tb_writer.close()

    return global_step


def evaluate_dev(args, model, passage_cache, source=""):

    dev_query_collection_path = os.path.join(args.data_dir, "dev-query{}".format(source))
    dev_query_cache = EmbeddingCache(dev_query_collection_path)

    logger.info('NLL validation ...')
    model.eval()

    log_result_step = 100
    batches = 0
    total_loss = 0.0
    total_correct_predictions = 0

    with dev_query_cache:
        dev_data_path = os.path.join(args.data_dir, "dev-data{}".format(source))
        with open(dev_data_path, 'r') as f:
            dev_data = f.readlines()
        dev_dataset = StreamingDataset(dev_data, GetTrainingDataProcessingFn(args, dev_query_cache, passage_cache, shuffle=False))
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.train_batch_size*2)

        for i, batch in enumerate(dev_dataloader):
            loss, correct_cnt = do_biencoder_fwd_pass(args, model, batch)
            loss.backward() # get CUDA oom without this
            model.zero_grad()
            total_loss += loss.item()
            total_correct_predictions += correct_cnt
            batches += 1
            if (i + 1) % log_result_step == 0:
                logger.info('Eval step: %d , loss=%f ', i, loss.item())

    total_loss = total_loss / batches
    total_samples = batches * args.train_batch_size * torch.distributed.get_world_size()
    correct_ratio = float(total_correct_predictions / total_samples)
    logger.info('NLL Validation: loss = %f. correct prediction ratio  %d/%d ~  %f', total_loss,
                total_correct_predictions,
                total_samples,
                correct_ratio
                )

    model.train()
    return total_loss, correct_ratio


def triplet_fwd_pass(args, model, batch):
    batch = tuple(t.to(args.device) for t in batch)
    inputs = {"query_ids": batch[0].long(), "attention_mask_q": batch[1].long(), 
                "input_ids_a": batch[3].long(), "attention_mask_a": batch[4].long(),
                "input_ids_b": batch[6].long(), "attention_mask_b": batch[7].long()}
    loss = model(**inputs)[0]

    if args.n_gpu > 1:
        loss = loss.mean()
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    return loss


def do_biencoder_fwd_pass(args, model, batch) -> (
        torch.Tensor, int):

    batch = tuple(t.to(args.device) for t in batch)
    inputs = {"query_ids": batch[0][::2].long(), "attention_mask_q": batch[1][::2].long(), 
                "input_ids_a": batch[3].long(), "attention_mask_a": batch[4].long()}

    local_q_vector, local_ctx_vectors = model(**inputs)

    q_vector_to_send = torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
    ctx_vector_to_send = torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach_()

    global_question_ctx_vectors = all_gather_list(
        [q_vector_to_send, ctx_vector_to_send],
        max_size=150000)

    global_q_vector = []
    global_ctxs_vector = []

    for i, item in enumerate(global_question_ctx_vectors):
        q_vector, ctx_vectors = item

        if i != args.local_rank:
            global_q_vector.append(q_vector.to(local_q_vector.device))
            global_ctxs_vector.append(ctx_vectors.to(local_q_vector.device))
        else:
            global_q_vector.append(local_q_vector)
            global_ctxs_vector.append(local_ctx_vectors)

    global_q_vector = torch.cat(global_q_vector, dim=0)
    global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)

    scores = torch.matmul(global_q_vector, torch.transpose(global_ctxs_vector, 0, 1))
    if len(global_q_vector.size()) > 1:
        q_num = global_q_vector.size(0)
        scores = scores.view(q_num, -1)
    softmax_scores = F.log_softmax(scores, dim=1)
    positive_idx_per_question = [i*2 for i in range(q_num)]
    loss = F.nll_loss(softmax_scores, torch.tensor(positive_idx_per_question).to(softmax_scores.device),
                      reduction='mean')
    max_score, max_idxs = torch.max(softmax_scores, 1)
    correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

    is_correct = correct_predictions_count.sum().item()

    if args.n_gpu > 1:
        loss = loss.mean()
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    return loss, is_correct

def _save_checkpoint(args, model, optimizer, scheduler, step: int) -> str:
    offset = step
    epoch = 0
    model_to_save = get_model_obj(model)
    cp = os.path.join(args.output_dir, 'checkpoint-' + str(offset))

    meta_params = {}

    state = CheckpointState(model_to_save.state_dict(),
                            optimizer.state_dict(),
                            scheduler.state_dict(),
                            offset,
                            epoch, meta_params
                            )
    torch.save(state._asdict(), cp)
    logger.info('Saved checkpoint at %s', cp)
    return cp

def _load_saved_state(model, optimizer, scheduler, saved_state: CheckpointState):
    epoch = saved_state.epoch
    step = saved_state.offset
    logger.info('Loading checkpoint @ step=%s', step)

    model_to_load = get_model_obj(model)
    logger.info('Loading saved model state ...')
    model_to_load.load_state_dict(saved_state.model_dict)  # set strict=False if you use extra projection

    return step


def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the cached passage and query files",
    )
    parser.add_argument(
        "--ann_dir",
        default=None,
        type=str,
        required=True,
        help="The ann training data dir. Should contain the output of ann data generation job",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MSMarcoConfigDict.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--num_epoch",
        default=0,
        type=int,
        help="Number of epoch to train, if specified will use training data instead of ann",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
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
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument("--triplet", default = False, action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="Tensorboard log dir",
    )

    parser.add_argument(
        "--optimizer",
        default="adamW",
        type=str,
        help="Optimizer - lamb or adamW",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=2.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=300000,
        type=int,
        help="If > 0: set total number of training steps to perform",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

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

    # ----------------- ANN HyperParam ------------------

    parser.add_argument(
        "--load_optimizer_scheduler",
        default = False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--single_warmup",
        default = True,
        action="store_true",
        help="use single or re-warmup",
    )

    # ----------------- End of Doc Ranking HyperParam ------------------
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    args = parser.parse_args()

    return args


def set_env(args):
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
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


def load_model(args):
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    args.output_mode = "classification"
    label_list = ["0", "1"]
    num_labels = len(label_list)


    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    configObj = MSMarcoConfigDict[args.model_type]
    tokenizer = configObj.tokenizer_class.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if is_first_worker():
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    model = configObj.model_class(args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    return tokenizer, model


def main():
    args = get_arguments()
    set_env(args)
    tokenizer, model = load_model(args)

    query_collection_path = os.path.join(args.data_dir, "train-query")
    query_cache = EmbeddingCache(query_collection_path)
    passage_collection_path = os.path.join(args.data_dir, "passages")
    passage_cache = EmbeddingCache(passage_collection_path)

    with query_cache, passage_cache:
        global_step = train(args, model, tokenizer, query_cache, passage_cache)
        logger.info(" global_step = %s", global_step)
    
    if args.local_rank != -1:
        dist.barrier()


if __name__ == "__main__":
    main()