import os
import sys
import time
import traceback
from tqdm import tqdm
import argparse
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import OpenMatch as om
from transformers import AdamW

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from utils import is_first_worker, DistributedEvalSampler, merge_resfile, set_dist_args, optimizer_to
from contextlib import nullcontext # from contextlib import suppress as nullcontext # for python < 3.7
torch.multiprocessing.set_sharing_strategy('file_system')
import logging
import random
import numpy as np

from tensorboardX import SummaryWriter
from magic_module import MagicModule

logger = logging.getLogger(__name__)


def dev(args, model, metric, dev_loader, device, global_step):
    torch.cuda.empty_cache()
    logger.info("start eval {} step ...".format(global_step))
    model.eval()
    rst_dict = {}
    for idx, dev_batch in enumerate(tqdm(dev_loader)):
        query_id, doc_id, label, retrieval_score = dev_batch['query_id'], dev_batch['doc_id'], dev_batch['label'], dev_batch['retrieval_score']
        with torch.no_grad():
            if args.model == 'bert':
                batch_score, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device), dev_batch['segment_ids'].to(device))
            if args.task == 'classification':
                batch_score = batch_score.softmax(dim=-1)[:, 1].squeeze(-1)
            batch_score = batch_score.detach().cpu().tolist()
            for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
                if q_id not in rst_dict:
                    rst_dict[q_id] = {}
                if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                    rst_dict[q_id][d_id] = [b_s, l]
    torch.cuda.empty_cache()
    return rst_dict


class RecurrDataLoader(object):
    def __init__(
        self,
        target_loader,
    ):
        self.target_loader = target_loader
        self.reset_iter()

    def reset_iter(self):
        self.target_iterator = iter(self.target_loader)

    def gen_target_inputs(self):
        try:
            target_batch = next(self.target_iterator)
        except StopIteration:
            self.reset_iter()
            target_batch = next(self.target_iterator)
        return target_batch


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def convert2delta(meta_model, grads, lr):
    names2deltas = {}
    for n, p in meta_model.named_buffers():
        if "pooler" not in n:
            names2deltas[n] = - lr * grads[len(names2deltas)]
    return names2deltas


def convert_to_cuda(input_batch, device):
    for key, value in input_batch.items():
        input_batch[key] = value.to(device)
    return input_batch


def mkdir_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def save_checkpoint(save_dir, model, m_optim, m_scheduler, global_step):
    # save model
    torch.save(
        model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), 
        os.path.join(save_dir, "model.bin")
    )
    # save optimizer
    torch.save(
        m_optim.state_dict(),
        os.path.join(save_dir, "optimizer.pt")
    )

    # save scheduler
    torch.save(
        m_scheduler.state_dict(),
        os.path.join(save_dir, "scheduler.pt")
    )
    logger.info("success save checkpoint-{} step!".format(global_step))


if __name__ == "__main__":

    ## *****************************
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='ranking')
    parser.add_argument('-ranking_loss', type=str, default='margin_loss')
    parser.add_argument('-model', type=str, default='bert')
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-reinfoselect', action='store_true', default=False)
    parser.add_argument('-eval_during_train', action='store_true', default=False)
    parser.add_argument('-reset', action='store_true', default=False)
    parser.add_argument('-train', action=om.utils.DictOrStr, default='./data/train_toy.jsonl')
    parser.add_argument('-max_input', type=int, default=1280000)

    parser.add_argument('-dev', action=om.utils.DictOrStr, default='./data/dev_toy.jsonl')
    parser.add_argument('-target', action=om.utils.DictOrStr, default=None)
    parser.add_argument('-qrels', type=str, default='./data/qrels_toy')
    parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-ent_vocab', type=str, default='')
    parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')

    parser.add_argument('-metric', type=str, default='ndcg_cut_10')
    parser.add_argument('-mode', type=str, default='cls')
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-max_query_len', type=int, default=20)
    parser.add_argument('-max_doc_len', type=int, default=150)
    parser.add_argument('-maxp', action='store_true', default=False)
    parser.add_argument('-epoch', type=int, default=1)

    parser.add_argument('-train_batch_size', type=int, default=8)
    parser.add_argument('-target_batch_size', type=int, default=8)
    parser.add_argument('-dev_eval_batch_size', type=int, default=128)
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-n_warmup_steps', type=int, default=1000)
    parser.add_argument('-gradient_accumulation_steps', type=int, default=4)
    parser.add_argument("-max_grad_norm", default=1.0,type=float, help="Max gradient norm.",)
    parser.add_argument('-eval_every', type=int, default=1000)
    parser.add_argument('-logging_step', type=int, default=100)
    parser.add_argument('-test_init_log', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=-1) # for distributed mode
    parser.add_argument( "--server_ip",type=str,default="", help="For distant debugging.",)
    parser.add_argument( "--server_port",type=str, default="",help="For distant debugging.",)

    parser.add_argument('-job_name', type=str, required=True)
    parser.add_argument('-save_folder', type=str, required=True)
    parser.add_argument('-log_weights', action='store_true', default=False)
    parser.add_argument('-checkpoint', type=str, default=None)
    parser.add_argument('-skip_trained_step', action='store_true', default=False)

    args = parser.parse_args()
    set_dist_args(args) # get local cpu/gpu device

    ## *************************************
    ## Save Dir
    ## *************************************
    mkdir_folder(args.save_folder) ## 1-level folder
    args.save_folder = os.path.join(args.save_folder, args.job_name + "_" + time.strftime("%m%d-%H%M-%S"))
    mkdir_folder(args.save_folder) ## 2-level folder

    args.tensorboard_dir = os.path.join(args.save_folder, "tensorboard")
    args.checkpoint_dir = os.path.join(args.save_folder, "checkpoints")
    mkdir_folder(args.tensorboard_dir)
    mkdir_folder(args.checkpoint_dir)

    ## *************************************
    # logging file
    ## *************************************
    args.log_file = os.path.join(args.save_folder, "logging.txt")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s: [ %(message)s ]", "%m/%d/%Y %I:%M:%S %p")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        logfile = logging.FileHandler(args.log_file, "w")
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info("COMMAND: %s" % " ".join(sys.argv))

    ## tensorboard
    writer = SummaryWriter(args.tensorboard_dir)

    ## *************************************
    ## Loading Dataset
    ## *************************************
    tokenizer = AutoTokenizer.from_pretrained(args.vocab)

    ## training data
    logger.info('reading training data...')
    train_set = om.data.datasets.MetaBertDataset(
        dataset=args.train,
        tokenizer=tokenizer,
        mode='train',
        query_max_len=args.max_query_len,
        doc_max_len=args.max_doc_len,
        max_input=args.max_input,
        task=args.task
    )
    
    ## dev data
    logger.info('reading dev data...')
    dev_set = om.data.datasets.MetaBertDataset(
        dataset=args.dev,
        tokenizer=tokenizer,
        mode='dev',
        query_max_len=args.max_query_len,
        doc_max_len=args.max_doc_len,
        max_input=args.max_input,
        task=args.task,
        _docs=train_set._docs,
    )

    ## target data
    logger.info('reading target data...')
    target_set = om.data.datasets.MetaBertDataset(
        dataset=args.target,
        tokenizer=tokenizer,
        mode='target',
        query_max_len=args.max_query_len,
        doc_max_len=args.max_doc_len,
        max_input=args.max_input,
        task=args.task,
        _docs=dev_set._docs,
        _queries=dev_set._queries,
    )

    ## dataloader
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_set)
        train_loader = om.data.DataLoader(
            dataset=train_set,
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=4,
            sampler=train_sampler
        )

        dev_sampler = DistributedEvalSampler(dev_set)
        dev_loader = om.data.DataLoader(
            dataset=dev_set,
            batch_size=args.dev_eval_batch_size,
            shuffle=False,
            num_workers=8,
            sampler=dev_sampler
        )
        target_sampler = DistributedSampler(target_set)
        target_loader = om.data.DataLoader(
            dataset=target_set,
            batch_size=args.target_batch_size,
            shuffle=False,
            num_workers=4,
            sampler=target_sampler
        )

        iter_target_loader = RecurrDataLoader(target_loader)
        dist.barrier()

    else:
        train_loader = om.data.DataLoader(
            dataset=train_set,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=4,
        )
        dev_loader = om.data.DataLoader(
            dataset=dev_set,
            batch_size=args.dev_eval_batch_size,
            shuffle=False,
            num_workers=12
        )
        target_loader = om.data.DataLoader(
            dataset=target_set,
            batch_size=args.target_batch_size,
            shuffle=True,
            num_workers=4,
        )
        iter_target_loader = RecurrDataLoader(target_loader)

    ## *************************************
    ## Model
    ## *************************************
    args.model = args.model.lower()
    model = om.models.Bert(
        pretrained=args.pretrain,
        mode=args.mode,
        task=args.task
    )

    ## *************************************
    ## Load Checkpoint
    ## *************************************
    steps_trained_in_current_epoch = 0
    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint)
        st = {}
        for k in state_dict:
            if k.startswith('bert'):
                st['_model'+k[len('bert'):]] = state_dict[k]
            elif k.startswith('classifier'):
                st['_dense'+k[len('classifier'):]] = state_dict[k]
            else:
                st[k] = state_dict[k]
        model.load_state_dict(st)
        logger.info("success load checkpoint from {}".format(args.checkpoint))

        ## Petrained steps
        have_updates = int(args.checkpoint.split("-")[-1].strip(".bin"))

        if args.skip_trained_step:
            steps_trained_in_current_epoch = int(args.checkpoint.split("-")[-1].strip(".bin")) * args.gradient_accumulation_steps
            logger.info("Training from {} steps".format(steps_trained_in_current_epoch))
    # set device
    model.to(args.device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        dist.barrier()

    model.zero_grad()
    model.train()

    ## *************************************
    ## Optimizer & Scheduler
    ## *************************************
    if args.optimizer.lower() == 'adam':
        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        m_optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    optimizer_to(m_optim, args.device)

    m_scheduler = get_linear_schedule_with_warmup(
        m_optim,
        num_warmup_steps=args.n_warmup_steps,
        num_training_steps=len(train_set)*args.epoch//args.train_batch_size
    )

    ## *************************************
    ## Loss
    ## *************************************
    if args.task == 'ranking':
        loss_fn = nn.MarginRankingLoss(margin=1, reduction='none')
    elif args.task == 'classification':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError('Task must be `ranking` or `classification`.')

    ## eval metric
    metric = om.metrics.Metric()

    # *********************************
    ## Start Training ...
    # *********************************
    train_loss = AverageMeter()
    target_loss = AverageMeter()
    stats = {'best_%s'%args.metric: 0.0, 'best_updates':0}

    global_step = 0
    for epoch in range(args.epoch):

        if args.local_rank != -1:
            train_sampler.set_epoch(epoch) # shuffle data for distributed
            logger.warning("current gpu local_rank {}".format(args.local_rank))

        for step, train_batch in enumerate(tqdm(train_loader, disable=args.local_rank not in [-1, 0])):

            sync_context = model.no_sync if (args.local_rank != -1 and (step+1) % args.gradient_accumulation_steps != 0) else nullcontext

            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            train_batch = convert_to_cuda(train_batch, args.device)

            # *********************************
            # [0] initialize a dummy model
            meta_model = MagicModule(model.module)
            meta_model.zero_grad()
            meta_model.train()

            # *********************************
            # [1] forward pass to compute the initial weighted loss
            # *********************************
            batch_score_pos, _ = meta_model(train_batch['input_ids_pos'], train_batch['input_mask_pos'], train_batch['segment_ids_pos'])
            batch_score_neg, _ = meta_model(train_batch['input_ids_neg'], train_batch['input_mask_neg'], train_batch['segment_ids_neg'])
            batch_loss = loss_fn(batch_score_pos.tanh(), batch_score_neg.tanh(), torch.ones(batch_score_pos.size()).to(args.device))

            eps = torch.zeros(batch_loss.size(), requires_grad=True).to(args.device)

            # init-weighted loss
            l_f_meta = torch.sum(batch_loss * eps)

            # *********************************
            # [2] meta-forward update
            # *********************************
            grads = torch.autograd.grad(l_f_meta, [p for n, p in meta_model.named_buffers() if "pooler" not in n], create_graph=True)
            names2deltas = convert2delta(meta_model, grads, lr=m_scheduler.get_last_lr()[0])
            # pesudo-update
            meta_model.update_params(names2deltas)

            # *********************************
            # [3] meta-backward update
            # *********************************
            target_batch = iter_target_loader.gen_target_inputs()
            target_batch = convert_to_cuda(target_batch, args.device)

            target_batch_score_pos, _ = meta_model(target_batch['input_ids_pos'], target_batch['input_mask_pos'], target_batch['segment_ids_pos'])
            target_batch_score_neg, _ = meta_model(target_batch['input_ids_neg'], target_batch['input_mask_neg'], target_batch['segment_ids_neg'])
            target_batch_loss = loss_fn(target_batch_score_pos.tanh(), target_batch_score_neg.tanh(), torch.ones(target_batch_score_pos.size()).to(args.device))

            # target avg loss
            l_g_meta = torch.mean(target_batch_loss)
            grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

            # *********************************
            # [4] weight normalization
            # *********************************
            w_tilde = torch.clamp(-grad_eps, min=0)
            norm_c = torch.sum(w_tilde)

            if norm_c != 0:
                w = w_tilde / norm_c
            else:
                w = w_tilde

            # *********************************
            # [5] Actual Update
            # *********************************
            model.train()

            with sync_context():
                batch_score_pos, _ = model(train_batch['input_ids_pos'], train_batch['input_mask_pos'], train_batch['segment_ids_pos'])
                batch_score_neg, _ = model(train_batch['input_ids_neg'], train_batch['input_mask_neg'], train_batch['segment_ids_neg'])
                batch_loss = loss_fn(batch_score_pos.tanh(), batch_score_neg.tanh(), torch.ones(batch_score_pos.size()).to(args.device))

            # re-weighted loss
            l_f = torch.sum(batch_loss * w)

            if args.gradient_accumulation_steps > 1:
                l_f = l_f / args.gradient_accumulation_steps

            with sync_context():
                l_f.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                m_optim.step()
                m_scheduler.step()

                m_optim.zero_grad()
                model.zero_grad()
                global_step += 1

            # *********************************
            # Complete training step
            # *********************************
            l_f_ = l_f.item()
            l_g_meta_ = l_g_meta.item()
            w_ = w.detach().cpu().tolist()

            train_loss.update(l_f_)
            target_loss.update(l_g_meta_)

            ## logging weights
            if args.log_weights:
                with open(os.path.join(args.save_folder, "weights.txt"), 'a+', encoding="utf-8") as f:
                    line = str(global_step) + "\t" + "\t".join([str(w) for w in w_]) + "\n"
                    f.write(line)

            ## Logging Loss
            if (step + 1) % int(args.logging_step * args.gradient_accumulation_steps) == 0 and args.local_rank in [-1,0]:
                writer.add_scalar('Training/loss', train_loss.avg, global_step)
                writer.add_scalar('Target/loss', target_loss.avg, global_step)
                writer.add_scalar('lr', m_scheduler.get_last_lr()[0], global_step)
                train_loss.reset()
                target_loss.reset()

            ## Dev evaluation
            if (step + 1) % int(args.eval_every * args.gradient_accumulation_steps) == 0:
                if args.eval_during_train:
                    rst_dict = dev(args, model, metric, dev_loader, args.device, global_step)

                    if args.local_rank != -1:
                        # distributed mode, save dicts and merge
                        om.utils.save_trec(
                            os.path.join(args.save_folder, "latest_dev.trec") + "_rank_{:03}".format(args.local_rank),
                            rst_dict
                        )
                        dist.barrier()
                        if args.local_rank == 0:
                            merge_resfile(os.path.join(args.save_folder, "latest_dev.trec") + "_rank_*",
                                          os.path.join(args.save_folder, "latest_dev.trec")
                                         )
                    else:
                        om.utils.save_trec(os.path.join(args.save_folder, "latest_dev.trec"), rst_dict)

                    if args.local_rank in [-1,0]:
                        if args.metric.split('_')[0] == 'mrr':
                            mes = metric.get_mrr(args.qrels, os.path.join(args.save_folder, "latest_dev.trec"), args.metric)
                        else:
                            mes = metric.get_metric(args.qrels, os.path.join(args.save_folder, "latest_dev.trec"), args.metric)
                        writer.add_scalar('Dev/%s'%args.metric, mes, global_step)

                        if mes >= stats['best_%s'%args.metric]:
                            stats['best_%s'%args.metric] = mes
                            stats['best_updates'] = global_step
                            logger.info( "Update! || best %s : %.4f || global step: %d \n"
                                        %(args.metric, stats['best_%s'%args.metric], stats['best_updates']))
                            # rename latest.trec to best.trec
                            os.rename(os.path.join(args.save_folder, "latest_dev.trec"),
                                      os.path.join(args.save_folder, "best_dev.trec"))
                            try:
                                save_checkpoint(args.checkpoint_dir, model, m_optim, m_scheduler, global_step)
                            except:
                                continue
                else:
                    try:
                        subset_checkpoint_dir = os.path.join(args.checkpoint_dir, "step-{}".format(global_step))
                        mkdir_folder(subset_checkpoint_dir)
                        save_checkpoint(subset_checkpoint_dir, model, m_optim, m_scheduler, global_step)
                    except:
                        logging.error(str(traceback.format_exc()))
                        continue
