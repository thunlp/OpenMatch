import argparse

import torch
import torch.nn as nn
from torch.distributions import Categorical

import OpenMatch as om

def dev(args, model, metric, dev_loader, device):
    rst_dict = {}
    for dev_batch in dev_loader:
        query_id, doc_id, label, retrieval_score = dev_batch['query_id'], dev_batch['doc_id'], dev_batch['label'], dev_batch['retrieval_score']
        with torch.no_grad():
            if args.model == 'bert':
                batch_score, _ = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device), dev_batch['segment_ids'].to(device))
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
            for (q_id, d_id, l, b_s) in zip(query_id, doc_id, label, batch_score):
                if q_id in rst_dict:
                    rst_dict[q_id].append((l, b_s, d_id))
                else:
                    rst_dict[q_id] = [(l, b_s, d_id)]
    return rst_dict

def train_reinfoselect(args, model, policy, loss_fn, m_optim, p_optim, metric, train_loader, dev_loader, device):
    best_mes = 0.0
    with torch.no_grad():
        rst_dict = dev(args, model, metric, dev_loader, device)
        om.utils.save_trec(args.res, rst_dict)
        if args.metric.split('_')[0] == 'mrr':
            mes = metric.get_mrr(args.qrels, args.res, args.metric)
        else:
            mes = metric.get_metric(args.qrels, args.res, args.metric)
    if mes > best_mes:
        best_mes = mes
        print('save_model...')
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), args.save)
        else:
            torch.save(model.state_dict(), args.save)
    print('initial result: ', mes)
    last_mes = mes
    for epoch in range(args.epoch):
        avg_loss = 0.0
        for step, train_batch in enumerate(train_loader):
            model.zero_grad()
            policy.zero_grad()
            if args.model == 'bert':
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device), train_batch['segment_ids_pos'].to(device))
                    batch_score_pos, _ = model(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device), train_batch['segment_ids_pos'].to(device))
                    batch_score_neg, _ = model(train_batch['input_ids_neg'].to(device), train_batch['input_mask_neg'].to(device), train_batch['segment_ids_neg'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device))
                    batch_score, _ = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            elif args.model == 'edrm':
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                            train_batch['doc_pos_wrd_idx'].to(device), train_batch['doc_pos_wrd_mask'].to(device))
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
                    batch_probs, _ = policy(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                            train_batch['doc_wrd_idx'].to(device), train_batch['doc_wrd_mask'].to(device))
                    batch_score, _ = model(train_batch['query_wrd_idx'].to(device), train_batch['query_wrd_mask'].to(device),
                                           train_batch['doc_wrd_idx'].to(device), train_batch['doc_wrd_mask'].to(device),
                                           train_batch['query_ent_idx'].to(device), train_batch['query_ent_mask'].to(device),
                                           train_batch['doc_ent_idx'].to(device), train_batch['doc_ent_mask'].to(device),
                                           train_batch['query_des_idx'].to(device), train_batch['doc_des_idx'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            else:
                if args.task == 'ranking':
                    batch_probs, _ = policy(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                               train_batch['doc_pos_idx'].to(device), train_batch['doc_pos_mask'].to(device))
                    batch_score_pos, _ = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                               train_batch['doc_pos_idx'].to(device), train_batch['doc_pos_mask'].to(device))
                    batch_score_neg, _ = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                               train_batch['doc_neg_idx'].to(device), train_batch['doc_neg_mask'].to(device))
                elif args.task == 'classification':
                    batch_probs, _ = policy(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                           train_batch['doc_idx'].to(device), train_batch['doc_mask'].to(device))
                    batch_score, _ = model(train_batch['query_idx'].to(device), train_batch['query_mask'].to(device),
                                           train_batch['doc_idx'].to(device), train_batch['doc_mask'].to(device))
                else:
                    raise ValueError('Task must be `ranking` or `classification`.')
            dist = Categorical(batch_probs.softmax(dim=-1))
            action = dist.sample()
            mask = action.ge(0.5)
            weights = action.clone().detach().float().requires_grad_(False)
            log_prob_p = torch.masked_select(dist.log_prob(action), mask)
            log_prob_n = torch.masked_select(dist.log_prob(1-action), mask)

            if args.task == 'ranking':
                batch_loss = loss_fn(batch_score_pos.tanh(), batch_score_neg.tanh(), torch.ones(batch_score_pos.size()).to(device))
            elif args.task == 'classification':
                batch_loss = loss_fn(batch_score, train_batch['label'].to(device))
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
            if torch.cuda.device_count() > 1:
                batch_loss = batch_loss.mean(-1)
            batch_loss = batch_loss.mul(weights).mean()
            avg_loss += batch_loss.item()
            batch_loss.backward()
            m_optim.step()

            with torch.no_grad():
                rst_dict = dev(args, model, metric, dev_loader, device)
                om.utils.save_trec(args.res, rst_dict)
                if args.metric.split('_')[0] == 'mrr':
                    mes = metric.get_mrr(args.qrels, args.res, args.metric)
                else:
                    mes = metric.get_metric(args.qrels, args.res, args.metric)
            if mes > best_mes:
                best_mes = mes
                print('save_model...')
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), args.save)
                else:
                    torch.save(model.state_dict(), args.save)
            print(step+1, avg_loss, mes, best_mes)
            avg_loss = 0.0

            reward = mes - last_mes
            last_mes = mes
            if reward > 0:
                policy_loss = (-log_prob_p * reward).sum().unsqueeze(-1)
            else:
                policy_loss = (log_prob_n * reward).sum().unsqueeze(-1)
            policy_loss.backward()
            p_optim.step()

def train(args, model, loss_fn, m_optim, metric, train_loader, dev_loader, device):
    best_mes = 0.0
    for epoch in range(args.epoch):
        avg_loss = 0.0
        for step, train_batch in enumerate(train_loader):
            model.zero_grad()
            if args.model == 'bert':
                if args.task == 'ranking':
                    batch_score_pos, _ = model(train_batch['input_ids_pos'].to(device), train_batch['input_mask_pos'].to(device), train_batch['segment_ids_pos'].to(device))
                    batch_score_neg, _ = model(train_batch['input_ids_neg'].to(device), train_batch['input_mask_neg'].to(device), train_batch['segment_ids_neg'].to(device))
                elif args.task == 'classification':
                    batch_score, _ = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device))
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
                batch_loss = loss_fn(batch_score_pos.tanh(), batch_score_neg.tanh(), torch.ones(batch_score_pos.size()).to(device))
            elif args.task == 'classification':
                batch_loss = loss_fn(batch_score, train_batch['label'].to(device))
            else:
                raise ValueError('Task must be `ranking` or `classification`.')
            if torch.cuda.device_count() > 1:
                batch_loss = batch_loss.mean()
            avg_loss += batch_loss.item()
            batch_loss.backward()
            m_optim.step()

            if (step+1) % args.eval_every == 0:
                with torch.no_grad():
                    rst_dict = dev(args, model, metric, dev_loader, device)
                    om.utils.save_trec(args.res, rst_dict)
                    if args.metric.split('_')[0] == 'mrr':
                        mes = metric.get_mrr(args.qrels, args.res, args.metric)
                    else:
                        mes = metric.get_metric(args.qrels, args.res, args.metric)
                if mes > best_mes:
                    best_mes = mes
                    print('save_model...')
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), args.save)
                    else:
                        torch.save(model.state_dict(), args.save)
                print(step+1, avg_loss/args.eval_every, mes, best_mes)
                avg_loss = 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='ranking')
    parser.add_argument('-model', type=str, default='bert')
    parser.add_argument('-reinfoselect', action='store_true', default=False)
    parser.add_argument('-train', action=om.utils.DictOrStr, default='./data/train_toy.jsonl')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-save', type=str, default='./checkpoints/bert.bin')
    parser.add_argument('-dev', action=om.utils.DictOrStr, default='./data/dev_toy.jsonl')
    parser.add_argument('-qrels', type=str, default='./data/qrels_toy')
    parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-ent_vocab', type=str, default='')
    parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-res', type=str, default='./results/bert.trec')
    parser.add_argument('-metric', type=str, default='ndcg_cut_10')
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-max_query_len', type=int, default=20)
    parser.add_argument('-max_doc_len', type=int, default=150)
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-eval_every', type=int, default=1000)
    args = parser.parse_args()

    if args.model == 'bert':
        tokenizer = args.vocab
        print('reading training data...')
        train_set = om.data.datasets.BertDataset(
            dataset=args.train,
            tokenizer=tokenizer,
            mode='train',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
        print('reading dev data...')
        dev_set = om.data.datasets.BertDataset(
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

    train_loader = om.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8
    )
    dev_loader = om.data.DataLoader(
        dataset=dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8
    )

    if args.model.lower() == 'bert':
        model = om.models.Bert(
            pretrained=args.pretrain,
            enc_dim=768,
            task=args.task
        )
        if args.reinfoselect:
            policy = om.models.Bert(
                pretrained=args.pretrain,
                enc_dim=768,
                task='classification'
            )
    elif args.model.lower() == 'edrm':
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
    elif args.model.lower() == 'tk':
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
    elif args.model.lower() == 'cknrm':
        model = om.models.ConvKNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    elif args.model.lower() == 'knrm':
        model = om.models.KNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    else:
        raise ValueError('model name error.')

    if args.reinfoselect and args.model.lower() != 'bert':
        policy = om.models.ConvKNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            kernel_dim=128,
            kernel_sizes=[1, 2, 3],
            embed_matrix=tokenizer.get_embed_matrix(),
            task='classification'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.reinfoselect:
        if args.task == 'ranking':
            loss_fn = nn.MarginRankingLoss(margin=1, reduction='none')
        elif args.task == 'classification':
            loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            raise ValueError('Task must be `ranking` or `classification`.')
    else:
        if args.task == 'ranking':
            loss_fn = nn.MarginRankingLoss(margin=1)
        elif args.task == 'classification':
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError('Task must be `ranking` or `classification`.')
    m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    if args.reinfoselect:
        p_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=args.lr)
    metric = om.metrics.Metric()

    model.to(device)
    if args.reinfoselect:
        policy.to(device)
    loss_fn.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        loss_fn = nn.DataParallel(loss_fn)

    if args.reinfoselect:
        train_reinfoselect(args, model, policy, loss_fn, m_optim, p_optim, metric, train_loader, dev_loader, device)
    else:
        train(args, model, loss_fn, m_optim, metric, train_loader, dev_loader, device)

if __name__ == "__main__":
    main()
