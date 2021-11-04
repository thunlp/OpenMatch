import argparse
from random import shuffle
import torch
import torch.nn as nn
import os 
from transformers import AutoTokenizer
import OpenMatch as om
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from utils import DistributedEvalSampler
import torch.multiprocessing as mp
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='ranking')
    parser.add_argument('-model', type=str, default='bert')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-test', action=om.utils.DictOrStr, default='./data/test_toy.jsonl')
    parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-ent_vocab', type=str, default='')
    parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-checkpoint', type=str, default='./checkpoints/bert.bin')
    parser.add_argument('-res', type=str, default='./results/bert.trec')
    parser.add_argument('-mode', type=str, default='cls')
    parser.add_argument('-n_kernels', type=int, default=21)
    parser.add_argument('-max_query_len', type=int, default=32)
    parser.add_argument('-max_doc_len', type=int, default=256)
    parser.add_argument('-maxp', action='store_true', default=False)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-pos_word',type=str,default=' relevant')
    parser.add_argument('-neg_word',type=str,default=' irrelevant')
    parser.add_argument('-template',type=str,default="<q> is [MASK] (relevant or irrelevant) to <d>")
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--soft_prompt", action="store_true")

    args = parser.parse_args()
    mp.set_sharing_strategy('file_system')
    seed=13
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
    torch.cuda.manual_seed_all(seed)
    args.world_size = torch.distributed.get_world_size()
    rank = dist.get_rank()
    print('world_size:{},local_rank:{},rank:{}'.format(args.world_size,args.local_rank,rank))
    
    ###############################################################################
    
    #################################################################################
    args.model = args.model.lower()
    if args.model == 'bert':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading test data...')
        if args.task.startswith("prompt"):
            pos_word_id = tokenizer(args.pos_word, add_special_tokens=False)["input_ids"]
            neg_word_id = tokenizer(args.neg_word, add_special_tokens=False)["input_ids"]
            #print(pos_word_id, neg_word_id)
            if len(neg_word_id) > 1 or len(pos_word_id) > 1:
                raise ValueError("Label words longer than 1 after tokenization")
            pos_word_id = pos_word_id[0]
            neg_word_id = neg_word_id[0]
            #tokenizer.add_tokens(["[SP1]", "[SP2]", "[SP3]", "[SP4]"], special_tokens=True)  # For continuous prompt

        if args.maxp:
            test_set = om.data.datasets.BertMaxPDataset(
                dataset=args.test,
                tokenizer=tokenizer,
                mode='test',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task
            )
        else:
            test_set = om.data.datasets.BertDataset(
                dataset=args.test,
                tokenizer=tokenizer,
                mode='test',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task,
                template=args.template
            )

    elif args.model=='t5':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading testing data...')
        test_set = om.data.datasets.t5Dataset(
                dataset=args.test,
                tokenizer=tokenizer,
                mode='test',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task,
                isv11=False
            )
    elif args.model=="simplet5":
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading testing data...')
        test_set = om.data.datasets.Simplet5Dataset(
                dataset=args.test,
                tokenizer=tokenizer,
                mode='test',
                query_max_len=args.max_query_len,
                doc_max_len=args.max_doc_len,
                max_input=args.max_input,
                task=args.task,
            )
    elif args.model == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        print('reading test data...')
        if args.task.startswith("prompt"):
            #print(args.neg_word,args.pos_word)
            pos_word_id = tokenizer(args.pos_word, add_special_tokens=False)["input_ids"]
            neg_word_id = tokenizer(args.neg_word, add_special_tokens=False)["input_ids"]
            #print(pos_word_id)
            #print(neg_word_id)
            if len(neg_word_id) > 1 or len(pos_word_id) > 1:
                raise ValueError("Label words longer than 1 after tokenization")
            pos_word_id = pos_word_id[0]
            neg_word_id = neg_word_id[0]
            #tokenizer.add_tokens(["[SP1]", "[SP2]", "[SP3]", "[SP4]"], special_tokens=True)
        test_set = om.data.datasets.RobertaDataset(
            dataset=args.test,
            tokenizer=tokenizer,
            mode='test',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task,
            template=args.template
        )
    elif args.model == 'edrm':
        tokenizer = om.data.tokenizers.WordTokenizer(
            pretrained=args.vocab
        )
        ent_tokenizer = om.data.tokenizers.WordTokenizer(
            vocab=args.ent_vocab
        )
        print('reading test data...')
        test_set = om.data.datasets.EDRMDataset(
            dataset=args.test,
            wrd_tokenizer=tokenizer,
            ent_tokenizer=ent_tokenizer,
            mode='test',
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
        print('reading test data...')
        test_set = om.data.datasets.Dataset(
            dataset=args.test,
            tokenizer=tokenizer,
            mode='test',
            query_max_len=args.max_query_len,
            doc_max_len=args.max_doc_len,
            max_input=args.max_input,
            task=args.task
        )
    test_sampler=DistributedEvalSampler(
        test_set,
        shuffle=False,
        num_replicas=args.world_size,
    	rank=rank
        )
    
    test_loader = om.data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,#
        sampler=test_sampler,#
        num_workers=20
    )
    
    dist.barrier()
     
    if args.model == 'bert' or args.model =="roberta":
        if args.maxp:
            model = om.models.BertMaxP(
                pretrained=args.pretrain,
                max_query_len=args.max_query_len,
                max_doc_len=args.max_doc_len,
                mode=args.mode,
                task=args.task
            )
        else:
            if args.task.startswith("prompt"):
                model = om.models.BertPrompt(
                    pretrained=args.pretrain,
                    mode=args.mode,
                    task=args.task,
                    pos_word_id=pos_word_id,
                    neg_word_id=neg_word_id,
                    soft_prompt=args.soft_prompt
                )
                model._model.resize_token_embeddings(len(tokenizer))
            else:
                model = om.models.Bert(
                    pretrained=args.pretrain,
                    mode=args.mode,
                    task=args.task
                )
    elif args.model =="t5":
        model = om.models.t5(checkpoint=args.pretrain)
    elif args.model =="simplet5":
        model = om.models.Simplet5(checkpoint=args.pretrain)
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

    model.to(device)
    model=torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True
        )
    dist.barrier()
###########################################################################################################################     
    
    model.eval()
    rst_dict = {}
    
    for test_batch in tqdm(test_loader,disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            query_id, doc_id,label = test_batch['query_id'], test_batch['doc_id'],test_batch['label']
            if args.model== 't5' or args.model=="simplet5":
                    batch_score=model(input_ids=test_batch['input_ids'].to(device), 
                    attention_mask=test_batch['attention_mask'].to(device),
                    labels= test_batch['labels'].to(device),
                    label=test_batch['label'].to(device),
                    isv11=False
                    )
            elif args.model == 'bert':
                if args.task.startswith("prompt"):
                    batch_score, _ = model(test_batch['input_ids'].to(device), test_batch['mask_pos'].to(device),test_batch['input_mask'].to(device), test_batch['segment_ids'].to(device))
                else:
                    batch_score, _ = model(test_batch['input_ids'].to(device), test_batch['input_mask'].to(device), test_batch['segment_ids'].to(device))
            elif args.model == 'roberta':
                if args.task.startswith("prompt"):
                    batch_score, _ = model(test_batch['input_ids'].to(device), test_batch['mask_pos'].to(device), test_batch['input_mask'].to(device))
                else:
                    batch_score, _ = model(test_batch['input_ids'].to(device), test_batch['input_mask'].to(device))

            elif args.model == 'edrm':
                batch_score, _ = model(test_batch['query_wrd_idx'].to(device), test_batch['query_wrd_mask'].to(device),
                                        test_batch['doc_wrd_idx'].to(device), test_batch['doc_wrd_mask'].to(device),
                                        test_batch['query_ent_idx'].to(device), test_batch['query_ent_mask'].to(device),
                                        test_batch['doc_ent_idx'].to(device), test_batch['doc_ent_mask'].to(device),
                                        test_batch['query_des_idx'].to(device), test_batch['doc_des_idx'].to(device))
            else:
                batch_score, _ = model(test_batch['query_idx'].to(device), test_batch['query_mask'].to(device),
                                        test_batch['doc_idx'].to(device), test_batch['doc_mask'].to(device))
            if args.task == 'classification' or args.task == "prompt_classification":
                batch_score = batch_score.softmax(dim=-1)[:, 1].squeeze(-1)
            elif args.task == "prompt_ranking":
                batch_score = batch_score[:, 0]
            batch_score = batch_score.detach().cpu().tolist()
            for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
                if q_id not in rst_dict:
                    rst_dict[q_id] = {}
                if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                    rst_dict[q_id][d_id] = [b_s, l]


    
    om.utils.save_trec(args.res, rst_dict)

def clean_up():
    dist.destroy_process_group()

def set_up(rank,world_size):
    
    dist.init_process_group(
        backend="nccl",
    rank=rank,
    world_size=world_size
    )
    torch.cuda.set_device(rank)

if __name__ == "__main__":
    main()