import argparse

import torch
import torch.nn as nn

from transformers import AutoTokenizer
import OpenMatch as om

def dev(args, model, dev_loader, device):
    features = []
    rst_dict = {}
    for dev_batch in dev_loader:
        query_id, doc_id, label, retrieval_score = dev_batch['query_id'], dev_batch['doc_id'], dev_batch['label'], dev_batch['retrieval_score']
        with torch.no_grad():
            if args.model == 'bert':
                batch_score, batch_feature = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device), dev_batch['segment_ids'].to(device))
            elif args.model == 'roberta':
                batch_score, batch_feature = model(dev_batch['input_ids'].to(device), dev_batch['input_mask'].to(device))
            elif args.model == 'edrm':
                batch_score, batch_feature = model(dev_batch['query_wrd_idx'].to(device), dev_batch['query_wrd_mask'].to(device),
                                                   dev_batch['doc_wrd_idx'].to(device), dev_batch['doc_wrd_mask'].to(device),
                                                   dev_batch['query_ent_idx'].to(device), dev_batch['query_ent_mask'].to(device),
                                                   dev_batch['doc_ent_idx'].to(device), dev_batch['doc_ent_mask'].to(device),
                                                   dev_batch['query_des_idx'].to(device), dev_batch['doc_des_idx'].to(device))
            else:
                batch_score, batch_feature = model(dev_batch['query_idx'].to(device), dev_batch['query_mask'].to(device),
                                                   dev_batch['doc_idx'].to(device), dev_batch['doc_mask'].to(device))
            if args.task == 'classification':
                batch_score = batch_score.softmax(dim=-1)[:, 1].squeeze(-1)
            
            batch_score = batch_score.detach().cpu().tolist()
            batch_feature = batch_feature.detach().cpu().tolist()
            retrieval_score = retrieval_score.tolist()
            for (q_id, d_id, l, b_s, b_f, r_s) in zip(query_id, doc_id, label, batch_score, batch_feature, retrieval_score):
                feature = []
                feature.append(str(l))
                feature.append('id:' + q_id)
                for i, fi in enumerate(b_f):
                    feature.append(str(i+1) + ':' + str(fi))
                feature.append(str(i+2) + ':' + str(b_s))
                feature.append(str(i+3) + ':' + str(r_s))
                features.append(' '.join(feature))

            for (q_id, d_id, b_s) in zip(query_id, doc_id, batch_score):
                if q_id not in rst_dict:
                    rst_dict[q_id] = {}
                if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                    rst_dict[q_id][d_id] = [b_s]
    return features, rst_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='ranking')
    parser.add_argument('-model', type=str, default='bert')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-dev', action=om.utils.DictOrStr, default='./data/dev_toy.jsonl')
    parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-ent_vocab', type=str, default='')
    parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-checkpoint', type=str, default='./checkpoints/bert.bin')
    parser.add_argument('-res', type=str, default='./features/bert-base_fusion_firstp_features')
    parser.add_argument('-mode', type=str, default='cls')
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-max_query_len', type=int, default=32)
    parser.add_argument('-max_doc_len', type=int, default=256)
    parser.add_argument('-maxp', action='store_true', default=False)
    parser.add_argument('-batch_size', type=int, default=32)
    args = parser.parse_args()

    args.model = args.model.lower()
    if args.model == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading dev data...')
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

    dev_loader = om.data.DataLoader(
        dataset=dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8
    )

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
        model = on.models.KNRM(
            vocab_size=tokenizer.get_vocab_size(),
            embed_dim=tokenizer.get_embed_dim(),
            kernel_num=args.n_kernels,
            embed_matrix=tokenizer.get_embed_matrix(),
            task=args.task
        )
    else:
        raise ValueError('model name error.')

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    features, rst_dict = dev(args, model, dev_loader, device)
    om.utils.save_features(args.res, features)
    om.utils.save_trec(args.res.replace("_features", ".trec").replace("features", "results"), rst_dict)

if __name__ == "__main__":
    main()
