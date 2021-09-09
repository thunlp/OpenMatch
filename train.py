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
logger = logging.getLogger(__name__)
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter(log_dir='logs')


def dev(args, model, metric, dev_loader, device):
    rst_dict = {}
    for dev_batch in dev_loader:
        query_id, doc_id, label, retrieval_score = dev_batch['query_id'], dev_batch['doc_id'], dev_batch['label'], dev_batch['retrieval_score']
        with torch.no_grad():
            if args.model == 'bert':
                batch_score, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device), dev_batch['segment_ids'].to(device))
            elif args.model == 'roberta':
                batch_score, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device))
            elif args.model == 'edrm':
                batch_score, _ = model(dev_batch['query_wrd_idx'].to(device), dev_batch['query_wrd_mask'].to(device),
                                       dev_batch['doc_wrd_idx'].to(device), dev_batch['doc_wrd_mask'].to(device),
                                       dev_batch['query_ent_idx'].to(device), dev_batch['query_ent_mask'].to(device),
                                       dev_batch['doc_ent_idx'].to(device), dev_batch['doc_ent_mask'].to(device),
                                       dev_batch['query_des_idx'].to(device), dev_batch['doc_des_idx'].to(device))
            else:
                batch_score, _ = model(dev_batch['query_idx'].to(device), dev_batch['query_mask'].to(device),
                                       dev_batch['doc_idx'].to(device), dev_batch['doc_mask'].to(device))
            if args.task == 'classification':
                batch_score = batch_score.softmax(dim=-1)[:, 1].squeeze(-1)
            batch_score = batch_score.detach().cpu().tolist()
            for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
                if q_id not in rst_dict:
                    rst_dict[q_id] = {}
                if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                    rst_dict[q_id][d_id] = [b_s, l]
    return rst_dict

def train_reinfoselect(args, model, policy, loss_fn, m_optim, m_scheduler, p_optim, metric, train_loader, dev_loader, device):
    best_mes = 0.0
    with torch.no_grad():
        rst_dict = dev(args, model, metric, dev_loader, device)
        om.utils.save_trec(args.res, rst_dict)
        if args.metric.split('_')[0] == 'mrr':
            mes = metric.get_mrr(args.qrels, args.res, args.metric)
        else:
            mes = metric.get_metric(args.qrels, args.res, args.metric)
    if mes >= best_mes:
        best_mes = mes
        print('save_model...')
        if args.n_gpu > 1:
            torch.save(model.module.state_dict(), args.save)
        else:
            torch.save(model.state_dict(), args.save)
    print('initial result: ', mes)
    last_mes = mes
    for epoch in range(args.epoch):
        avg_loss = 0.0
        log_prob_ps = []
        log_prob_ns = []
        for step, train_batch in enumerate(train_loader):
            if args.model == 'bert':
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device), train_batch['segment_ids_pos'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'roberta':
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'edrm':
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                            train_batch['doc_pos_wrd_idx'].to(device), train_batch['doc_pos_wrd_mask'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                            train_batch['doc_wrd_idx'].to(device), train_batch['doc_wrd_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            else:
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                               train_batch['doc_pos_idx'].to(device), train_batch['doc_pos_mask'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                           train_batch['doc_idx'].to(device), train_batch['doc_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            batch_probs = F.gumbel_softmax(batch_probs, tau=args.tau)
            m = Categorical(batch_probs)
            action = m.sample()
            if action.sum().item() < 1:
                #m_scheduler.step()
                if (step+1) % args.eval_every == 0 and len(log_prob_ps) > 0:
                    with torch.no_grad():
                        rst_dict = dev(args, model, metric, dev_loader, device)
                        om.utils.save_trec(args.res, rst_dict)
                        if args.metric.split('_')[0] == 'mrr':
                            mes = metric.get_mrr(args.qrels, args.res, args.metric)
                        else:
                            mes = metric.get_metric(args.qrels, args.res, args.metric)
                    if mes >= best_mes:
                        best_mes = mes
                        print('save_model...')
                        if args.n_gpu > 1:
                            torch.save(model.module.state_dict(), args.save)
                        else:
                            torch.save(model.state_dict(), args.save)
                    
                    print(step+1, avg_loss/len(log_prob_ps), mes, best_mes)
                    avg_loss = 0.0

                    reward = mes - last_mes
                    last_mes = mes
                    if reward >= 0:
                        policy_loss = [(-log_prob_p * reward).sum().unsqueeze(-1) for log_prob_p in log_prob_ps]
                    else:
                        policy_loss = [(log_prob_n * reward).sum().unsqueeze(-1) for log_prob_n in log_prob_ns]
                    policy_loss = torch.cat(policy_loss).sum()
                    policy_loss.backward()
                    p_optim.step()
                    p_optim.zero_grad()

                    if args.reset:
                        state_dict = torch.load(args.save)
                        model.load_state_dict(state_dict)
                        last_mes = best_mes
                    log_prob_ps = []
                    log_prob_ns = []
                continue

            filt = action.nonzero().squeeze(-1).cpu()
            if args.model == 'bert':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['input_ids_pos'].index_select(0, filt).to(device),
                                               train_batch['input_mask_pos'].index_select(0, filt).to(device),
                                               train_batch['segment_ids_pos'].index_select(0, filt).to(device))
                    batch_score_neg, _ = model(train_batch['input_ids_neg'].index_select(0, filt).to(device),
                                               train_batch['input_mask_neg'].index_select(0, filt).to(device),
                                               train_batch['segment_ids_neg'].index_select(0, filt).to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].index_select(0, filt).to(device),
                                           train_batch['input_mask'].index_select(0, filt).to(device),
                                           train_batch['segment_ids'].index_select(0, filt).to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'roberta':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['input_ids_pos'].index_select(0, filt).to(device), train_batch['input_mask_pos'].index_select(0, filt).to(device))
                    batch_score_neg, _ = model(train_batch['input_ids_neg'].index_select(0, filt).to(device), train_batch['input_mask_neg'].index_select(0, filt).to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].index_select(0, filt).to(device), train_batch['input_mask'].index_select(0, filt).to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'edrm':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['query_wrd_idx'].index_select(0, filt).to(device), train_batch['query_wrd_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_pos_wrd_idx'].index_select(0, filt).to(device), train_batch['doc_pos_wrd_mask'].index_select(0, filt).to(device),
                                               train_batch['query_ent_idx'].index_select(0, filt).to(device), train_batch['query_ent_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_pos_ent_idx'].index_select(0, filt).to(device), train_batch['doc_pos_ent_mask'].index_select(0, filt).to(device),
                                               train_batch['query_des_idx'].index_select(0, filt).to(device), train_batch['doc_pos_des_idx'].index_select(0, filt).to(device))
                    batch_score_neg, _ = model(train_batch['query_wrd_idx'].index_select(0, filt).to(device), train_batch['query_wrd_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_neg_wrd_idx'].index_select(0, filt).to(device), train_batch['doc_neg_wrd_mask'].index_select(0, filt).to(device),
                                               train_batch['query_ent_idx'].index_select(0, filt).to(device), train_batch['query_ent_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_neg_ent_idx'].index_select(0, filt).to(device), train_batch['doc_neg_ent_mask'].index_select(0, filt).to(device),
                                               train_batch['query_des_idx'].index_select(0, filt).to(device), train_batch['doc_neg_des_idx'].index_select(0, filt).to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['query_wrd_idx'].index_select(0, filt).to(device), train_batch['query_wrd_mask'].index_select(0, filt).to(device),
                                           train_batch['doc_wrd_idx'].index_select(0, filt).to(device), train_batch['doc_wrd_mask'].index_select(0, filt).to(device),
                                           train_batch['query_ent_idx'].index_select(0, filt).to(device), train_batch['query_ent_mask'].index_select(0, filt).to(device),
                                           train_batch['doc_ent_idx'].index_select(0, filt).to(device), train_batch['doc_ent_mask'].index_select(0, filt).to(device),
                                           train_batch['query_des_idx'].index_select(0, filt).to(device), train_batch['doc_des_idx'].index_select(0, filt).to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            else:
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['query_idx'].index_select(0, filt).to(device), train_batch['query_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_pos_idx'].index_select(0, filt).to(device), train_batch['doc_pos_mask'].index_select(0, filt).to(device))
                    batch_score_neg, _ = model(train_batch['query_idx'].index_select(0, filt).to(device), train_batch['query_mask'].index_select(0, filt).to(device),
                                               train_batch['doc_neg_idx'].index_select(0, filt).to(device), train_batch['doc_neg_mask'].index_select(0, filt).to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['query_idx'].index_select(0, filt).to(device), train_batch['query_mask'].index_select(0, filt).to(device),
                                           train_batch['doc_idx'].index_select(0, filt).to(device), train_batch['doc_mask'].index_select(0, filt).to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')

            mask = action.ge(0.5)
            log_prob_p = m.log_prob(action)
            log_prob_n = m.log_prob(1-action)
            log_prob_ps.append(torch.masked_select(log_prob_p, mask))
            log_prob_ns.append(torch.masked_select(log_prob_n, mask))

            if args.task == 'ranking':
                batch_loss = loss_fn(batch_score_pos.tanh(), batch_score_neg.tanh(), torch.ones(batch_score_pos.size()).to(device))
            elif args.task == 'classification':
                batch_loss = loss_fn(batch_score, train_batch['label'].to(device))
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
            
            if args.n_gpu > 1:
                batch_loss = batch_loss.mean(-1)
            batch_loss = batch_loss.mean()
            avg_loss += batch_loss.item()
            
            batch_loss.backward()
            m_optim.step()
            m_scheduler.step()
            m_optim.zero_grad()

            if (step+1) % args.eval_every == 0:
                with torch.no_grad():
                    rst_dict = dev(args, model, metric, dev_loader, device)
                    om.utils.save_trec(args.res, rst_dict)
                    if args.metric.split('_')[0] == 'mrr':
                        mes = metric.get_mrr(args.qrels, args.res, args.metric)
                    else:
                        mes = metric.get_metric(args.qrels, args.res, args.metric)
                if mes >= best_mes:
                    best_mes = mes
                    print('save_model...')
                    if args.n_gpu > 1:
                        torch.save(model.module.state_dict(), args.save)
                    else:
                        torch.save(model.state_dict(), args.save)
                print(step+1, avg_loss/len(log_prob_ps), mes, best_mes)
                avg_loss = 0.0

                reward = mes - last_mes
                last_mes = mes
                if reward >= 0:
                    policy_loss = [(-log_prob_p * reward).sum().unsqueeze(-1) for log_prob_p in log_prob_ps]
                else:
                    policy_loss = [(log_prob_n * reward).sum().unsqueeze(-1) for log_prob_n in log_prob_ns]
                policy_loss = torch.cat(policy_loss).sum()
                policy_loss.backward()
                p_optim.step()
                p_optim.zero_grad()

                if args.reset:
                    state_dict = torch.load(args.save)
                    model.load_state_dict(state_dict)
                    last_mes = best_mes
                log_prob_ps = []
                log_prob_ns = []

def train(args, model, loss_fn, m_optim, m_scheduler, metric, train_loader, dev_loader, device, train_sampler=None):
    best_mes = 0.0
    global_step = 0 # steps that outside epoches
    for epoch in range(args.epoch):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch) # shuffle data for distributed
            logger.warning("current gpu local_rank {}".format(args.local_rank))

        avg_loss = 0.0
        for step, train_batch in enumerate(train_loader):
            
            sync_context = model.no_sync if (args.local_rank != -1 and (step+1) % args.gradient_accumulation_steps != 0) else nullcontext

            if args.model == 'bert':
                if args.task == 'ranking':
                    # sync gradients only at gradient accumulation step
                    with sync_context():
                        batch_score_pos, _ = model(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device), train_batch['segment_ids_pos'].to(device))
                        batch_score_neg, _ = model(train_batch['input_ids_neg'].to(device), train_batch['input_mask_neg'].to(device), train_batch['segment_ids_neg'].to(device))
                elif args.task == 'classification':
                    with sync_context():
                        batch_score, _ = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'roberta':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device))
                    batch_score_neg, _ = model(train_batch['input_ids_neg'].to(device), train_batch['input_mask_neg'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'edrm':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                               train_batch['doc_pos_wrd_idx'].to(device), train_batch['doc_pos_wrd_mask'].to(device),
                                               train_batch['query_ent_idx'].to(device), train_batch['query_ent_mask'].to(device),
                                               train_batch['doc_pos_ent_idx'].to(device), train_batch['doc_pos_ent_mask'].to(device),
                                               train_batch['query_des_idx'].to(device), train_batch['doc_pos_des_idx'].to(device))
                    batch_score_neg, _ = model(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                               train_batch['doc_neg_wrd_idx'].to(device), train_batch['doc_neg_wrd_mask'].to(device),
                                               train_batch['query_ent_idx'].to(device), train_batch['query_ent_mask'].to(device),
                                               train_batch['doc_neg_ent_idx'].to(device), train_batch['doc_neg_ent_mask'].to(device),
                                               train_batch['query_des_idx'].to(device), train_batch['doc_neg_des_idx'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                           train_batch['doc_wrd_idx'].to(device), train_batch['doc_wrd_mask'].to(device),
                                           train_batch['query_ent_idx'].to(device), train_batch['query_ent_mask'].to(device),
                                           train_batch['doc_ent_idx'].to(device), train_batch['doc_ent_mask'].to(device),
                                           train_batch['query_des_idx'].to(device), train_batch['doc_des_idx'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            else:
                if args.task == 'ranking':
                    with sync_context():
                        batch_score_pos, _ = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                                train_batch['doc_pos_idx'].to(device), train_batch['doc_pos_mask'].to(device))
                        batch_score_neg, _ = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                                train_batch['doc_neg_idx'].to(device), train_batch['doc_neg_mask'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                           train_batch['doc_idx'].to(device), train_batch['doc_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')

            if args.task == 'ranking':
                with sync_context():
                    if args.ranking_loss == 'margin_loss':
                        batch_loss = loss_fn(batch_score_pos.tanh(), batch_score_neg.tanh(), torch.ones(batch_score_pos.size()).to(device))
                    elif args.ranking_loss == 'CE_loss':
                        batch_loss = loss_fn(torch.sigmoid(batch_score_pos-batch_score_neg),torch.ones(batch_score_neg.size()).to(device))
                    elif args.ranking_loss == 'triplet_loss':
                        logit_matrix = torch.cat([batch_score_pos.reshape([-1,1]),batch_score_neg.reshape([-1,1])], dim=1)
                        lsm = F.log_softmax(input=logit_matrix,dim=1)
                        batch_loss = torch.mean(-1.0 * lsm[:, 0])
                    elif args.ranking_loss == 'LCE_loss':
                        pass
            elif args.task == 'classification':
                with sync_context():
                    batch_loss = loss_fn(batch_score, train_batch['label'].to(device))
            else:
                raise ValueError('Task must be `ranking` or `classification`.')

            if args.n_gpu > 1:
                batch_loss = batch_loss.mean()
            if args.gradient_accumulation_steps > 1:
                batch_loss = batch_loss / args.gradient_accumulation_steps
            avg_loss += batch_loss.item()

            with sync_context():
                batch_loss.backward()
            # if args.local_rank != -1:
            #     if (step+1) % args.gradient_accumulation_steps == 0:
            #         batch_loss.backward()
            #     else:
            #         with model.no_sync():
            #             batch_loss.backward()
            # else:
            #     batch_loss.backward()

            if (step+1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                m_optim.step()
                m_scheduler.step()
                m_optim.zero_grad()
                global_step += 1

                if args.logging_step > 0 and ((global_step+1) % args.logging_step == 0 or (args.test_init_log and global_step==0)):
                    # if is_first_worker():
                    if args.local_rank in [-1,0]:
                        logger.info( "training gpu {}:,  global step: {}, local step: {}, loss: {}".format(args.local_rank,global_step+1, step+1, avg_loss/args.logging_step))
                        # writer.add_scalar('avg_loss',avg_loss/args.logging_step, step)
                        # writer.add_scalar('dev', mes, step)
                    avg_loss = 0.0 

                if (global_step+1) % args.eval_every == 0 or (args.test_init_log and global_step==0):                
                    model.eval()
                    with torch.no_grad():
                        rst_dict = dev(args, model, metric, dev_loader, device)
                    model.train()

                    if args.local_rank != -1:
                        # distributed mode, save dicts and merge
                        om.utils.save_trec(args.res + "_rank_{:03}".format(args.local_rank), rst_dict)
                        dist.barrier()
                        # if is_first_worker():
                        if args.local_rank in [-1,0]:
                            merge_resfile(args.res + "_rank_*", args.res)

                    else:
                        om.utils.save_trec(args.res, rst_dict)
                        
                    # if is_first_worker():
                    if args.local_rank in [-1,0]:
                        if args.metric.split('_')[0] == 'mrr':
                            mes = metric.get_mrr(args.qrels, args.res, args.metric)
                        else:
                            mes = metric.get_metric(args.qrels, args.res, args.metric)

                        best_mes = mes if mes >= best_mes else best_mes
                        logger.info( 'save_model at step {}'.format(global_step+1))
                        if args.n_gpu > 1:
                            torch.save(model.module.state_dict(), args.save + "_step-{}".format(global_step+1))
                        else:
                            torch.save(model.state_dict(), args.save + "_step-{}".format(global_step+1))
                        logger.info( "global step: {}, messure: {}, best messure: {}".format(global_step+1, mes, best_mes))
            # dist.barrier()  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='ranking')
    parser.add_argument('-ranking_loss', type=str, default='margin_loss')
    parser.add_argument('-model', type=str, default='bert')
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-reinfoselect', action='store_true', default=False)
    parser.add_argument('-reset', action='store_true', default=False)
    parser.add_argument('-train', action=om.utils.DictOrStr, default='./data/train_toy.jsonl')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-save', type=str, default='./checkpoints/bert.bin')
    parser.add_argument('-dev', action=om.utils.DictOrStr, default='./data/dev_toy.jsonl')
    parser.add_argument('-qrels', type=str, default='./data/qrels_toy')
    parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-ent_vocab', type=str, default='')
    parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-checkpoint', type=str, default=None)
    parser.add_argument('-res', type=str, default='./results/bert.trec')
    parser.add_argument('-metric', type=str, default='ndcg_cut_10')
    parser.add_argument('-mode', type=str, default='cls')
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-max_query_len', type=int, default=20)
    parser.add_argument('-max_doc_len', type=int, default=150)
    parser.add_argument('-maxp', action='store_true', default=False)
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-dev_eval_batch_size', type=int, default=128)
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-n_warmup_steps', type=int, default=1000)
    parser.add_argument('-gradient_accumulation_steps', type=int, default=4) 
    parser.add_argument("-max_grad_norm", default=1.0,type=float,help="Max gradient norm.",)
    parser.add_argument('-eval_every', type=int, default=1000)
    parser.add_argument('-logging_step', type=int, default=100)
    parser.add_argument('-test_init_log', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=-1) # for distributed mode
    parser.add_argument( "--server_ip",type=str,default="", help="For distant debugging.",)  
    parser.add_argument( "--server_port",type=str, default="",help="For distant debugging.",)

    args = parser.parse_args()

    set_dist_args(args) # get local cpu/gpu device

    args.model = args.model.lower()
    if args.model == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        logger.info('reading training data...')
        if args.maxp:
            train_set = om.data.datasets.BertMaxPDataset(
                dataset=args.train,
                tokenizer=tokenizer,
                mode='train',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        else:
            train_set = om.data.datasets.BertDataset(
                dataset=args.train,
                tokenizer=tokenizer,
                mode='train',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        logger.info('reading dev data...')
        if args.maxp:
            dev_set = om.data.datasets.BertMaxPDataset(
                dataset=args.dev,
                tokenizer=tokenizer,
                mode='dev',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        else:
            dev_set = om.data.datasets.BertDataset(
                dataset=args.dev,
                tokenizer=tokenizer,
                mode='dev',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
               task=args.task
            )
    elif args.model == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading training data...')
        train_set = om.data.datasets.RobertaDataset(
            dataset=args.train,
            tokenizer=tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        dev_set = om.data.datasets.RobertaDataset(
            dataset=args.dev,
            tokenizer=tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
    elif args.model == 'edrm':
        tokenizer = om.data.tokenizers.WordTokenizer(
            pretrained=args.vocab
        )
        ent_tokenizer = om.data.tokenizers.WordTokenizer(
            vocab=args.ent_vocab
        )
        print('reading training data...')
        train_set = om.data.datasets.EDRMDataset(
            dataset=args.train,
            wrd_tokenizer=tokenizer,
            ent_tokenizer=ent_tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            des_max_len=20,
            max_ent_num=3,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        dev_set = om.data.datasets.EDRMDataset(
            dataset=args.dev,
            wrd_tokenizer=tokenizer,
            ent_tokenizer=ent_tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            des_max_len=20,
            max_ent_num=3,
            max_input=args.max_input,
            task=args.task
        )
    else:
        tokenizer = om.data.tokenizers.WordTokenizer(
            pretrained=args.vocab
        )
        print('reading training data...')
        train_set = om.data.datasets.Dataset(
            dataset=args.train,
            tokenizer=tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        dev_set = om.data.datasets.Dataset(
            dataset=args.dev,
            tokenizer=tokenizer,
            mode='dev',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )

    if args.local_rank != -1:
        # train_sampler = DistributedSampler(train_set, args.world_size, args.local_rank)
        train_sampler = DistributedSampler(train_set)
        train_loader = om.data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            sampler=train_sampler
        )
        #dev_sampler = DistributedSampler(dev_set)
        dev_sampler = DistributedEvalSampler(dev_set)
        dev_loader = om.data.DataLoader(
            dataset=dev_set,
            batch_size=args.batch_size * 16 if args.dev_eval_batch_size <= 0 else args.dev_eval_batch_size,
            shuffle=False,
            num_workers=1,
            sampler=dev_sampler
        )
        dist.barrier()

    else:
        train_loader = om.data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8
        )
        dev_loader = om.data.DataLoader(
            dataset=dev_set,
            batch_size=args.batch_size * 16,
            shuffle=False,
            num_workers=8
        )
        train_sampler = None

    if args.model == 'bert' or args.model == 'roberta':
        if args.maxp:
            model = om.models.BertMaxP(
                pretrained=args.pretrain,
                max_query_len=args.max_query_len,
                max_doc_len=args.max_doc_len,
                mode=args.mode,
                task=args.task
            )
        else:
            model = om.models.Bert(
                pretrained=args.pretrain,
                mode=args.mode,
                task=args.task
            )
        if args.reinfoselect:
            policy = om.models.Bert(
                pretrained=args.pretrain,
                mode=args.mode,
                task='classification'
            )
    elif args.model == 'edrm':
        model = om.models.EDRM(
            wrd_vocab_size=tokenizer.get_vocab_size(),
            ent_vocab_size=ent_tokenizer.get_vocab_size(),
            wrd_embed_dim=tokenizer.get_embed_dim(),
            ent_embed_dim=128,
            max_des_len=20,
            max_ent_num=3,
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            wrd_embed_matrix=tokenizer.get_embed_matrix(),
            ent_embed_matrix=None,
            task=args.task
        )
    elif args.model == 'tk':
        model = om.models.TK(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            head_num=10,
            hidden_dim=100,
            layer_num=2,
            kernel_num=args.n_kernels,
            dropout=0.0,
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    elif args.model == 'cknrm':
        model = om.models.ConvKNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    elif args.model == 'knrm':
        model = om.models.KNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    else:
        raise ValueError('model name error.')

    if args.reinfoselect and args.model != 'bert':
        policy = om.models.ConvKNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            embed_matrix=tokenizer.get_embed_matrix(),
            task='classification'
        )

    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint)
        if args.model == 'bert':
            st = {}
            for k in state_dict:
                if k.startswith('bert'):
                    st['_model'+k[len('bert'):]] = state_dict[k]
                elif k.startswith('classifier'):
                    st['_dense'+k[len('classifier'):]] = state_dict[k]
                else:
                    st[k] = state_dict[k]
            model.load_state_dict(st)
        else:
            model.load_state_dict(state_dict)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = args.device

    if args.reinfoselect:
        if args.task == 'ranking':
            loss_fn = nn.MarginRankingLoss(margin=1, reduction='none')
        elif args.task == 'classification':
            loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            raise ValueError('Task must be `ranking` or `classification`.')
    else:
        if args.task == 'ranking':
            if args.ranking_loss == 'margin_loss':
                loss_fn = nn.MarginRankingLoss(margin=1)
            elif args.ranking_loss == 'CE_loss':
                loss_fn = nn.BCELoss()
            elif args.ranking_loss == 'triplet_loss':
                loss_fn = nn.BCELoss() # dummpy loss for occupation
                # loss_fn = F.log_softmax(dim=1)
            elif args.ranking_loss == 'LCE_loss':
                print("LCE loss TODO")
                # nn.CrossEntropyLoss()

        elif args.task == 'classification':
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError('Task must be `ranking` or `classification`.')


    model.to(device)
    if args.reinfoselect:
        policy.to(device)
    loss_fn.to(device)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
        loss_fn = nn.DataParallel(loss_fn)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        dist.barrier()

    model.zero_grad()
    model.train()
    if args.optimizer.lower() == 'adam':
        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        m_optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if args.local_rank == -1:
        m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//args.batch_size)
    else:
        m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//(args.batch_size*args.world_size*args.gradient_accumulation_steps))
    if args.reinfoselect:
        p_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=args.lr)

    optimizer_to(m_optim,device)
    

    metric = om.metrics.Metric()

    logger.info(args)
    if args.reinfoselect:
        train_reinfoselect(args, model, policy, loss_fn, m_optim, m_scheduler, p_optim, metric, train_loader, dev_loader, device)
    else:
        train(args, model, loss_fn, m_optim, m_scheduler, metric, train_loader, dev_loader, device, train_sampler=train_sampler)

if __name__ == "__main__":
    main()