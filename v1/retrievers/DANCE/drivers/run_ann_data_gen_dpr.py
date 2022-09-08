import argparse
import sys
sys.path += ['../']
import json
import logging
import os
from os.path import isfile, join
import random
import time
import csv
import numpy as np
import torch
from torch.serialization import default_restore_location
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
from torch import nn
from model.models import MSMarcoConfigDict
from utils.util import (
    StreamingDataset, 
    EmbeddingCache, 
    get_checkpoint_no, 
    get_latest_ann_data,
    barrier_array_merge,
    is_first_worker,
    pickle_write,
)
from data.DPR_data import GetProcessingFn, load_mapping
from utils.dpr_utils import load_states_from_checkpoint, get_model_obj, SimpleTokenizer, has_answer
import random 
import transformers
from transformers import (
    AdamW,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer
)
from torch import nn
logger = logging.getLogger(__name__)
import faiss
import glob
from utils.indexing_utils import clean_faiss_gpu,get_gpu_index,document_split_faiss_index,loading_possitive_document_embedding
import pickle

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from utils.trec_convert import save_trec_file,convert_trec_to_MARCO_id

def get_latest_checkpoint(args):
    if not os.path.exists(args.training_dir):
        return args.init_model_dir, 0
    files = list(next(os.walk(args.training_dir))[2])

    def valid_checkpoint(checkpoint):
        return checkpoint.startswith("checkpoint-")

    logger.info("checkpoint files")
    logger.info(files)
    checkpoint_nums = [get_checkpoint_no(s) for s in files if valid_checkpoint(s)]

    if len(checkpoint_nums) > 0:
        return os.path.join(args.training_dir, "checkpoint-" + str(max(checkpoint_nums))), max(checkpoint_nums)
    return args.init_model_dir, 0


def load_data(args):
    passage_path = os.path.join(args.passage_path, "psgs_w100.tsv")
    test_qa_path = os.path.join(args.test_qa_path, "nq-test.csv")
    trivia_test_qa_path = os.path.join(args.trivia_test_qa_path, "trivia-test.csv")
    dev_qa_path = os.path.join(args.test_qa_path, "nq-dev.csv")
    trivia_dev_qa_path = os.path.join(args.trivia_test_qa_path, "trivia-dev.csv")
    train_ann_path = os.path.join(args.data_dir, "train-ann")

    pid2offset, offset2pid = load_mapping(args.data_dir, "pid2offset")

    passage_text = {}
    train_pos_id = []
    train_answers = []
    test_answers = []
    test_answers_trivia = []
    dev_answers = []
    dev_answers_trivia = []

    logger.info("Loading train ann")
    with open(train_ann_path, 'r', encoding='utf8') as f:
        # file format: q_id, positive_pid, answers
        tsvreader = csv.reader(f, delimiter="\t")
        for row in tsvreader:
            train_pos_id.append(int(row[1]))
            train_answers.append(eval(row[2]))

    logger.info("Loading test answers")
    with open(test_qa_path, "r", encoding="utf-8") as ifile:
        # file format: question, answers
        reader = csv.reader(ifile, delimiter='\t')
        for row in reader:
            test_answers.append(eval(row[1]))

    logger.info("Loading trivia test answers")
    with open(trivia_test_qa_path, "r", encoding="utf-8") as ifile:
        # file format: question, answers
        reader = csv.reader(ifile, delimiter='\t')
        for row in reader:
            test_answers_trivia.append(eval(row[1]))
    
    logger.info("Loading dev answers")
    with open(dev_qa_path, "r", encoding="utf-8") as ifile:
        # file format: question, answers
        reader = csv.reader(ifile, delimiter='\t')
        for row in reader:
            dev_answers.append(eval(row[1]))

    logger.info("Loading trivia dev answers")
    with open(trivia_dev_qa_path, "r", encoding="utf-8") as ifile:
        # file format: question, answers
        reader = csv.reader(ifile, delimiter='\t')
        for row in reader:
            dev_answers_trivia.append(eval(row[1]))

    logger.info("Loading passages")
    with open(passage_path, "r", encoding="utf-8") as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', )
        # file format: doc_id, doc_text, title
        for row in reader:
            if row[0] != 'id':
                passage_text[pid2offset[int(row[0])]] = (row[1], row[2])

    logger.info("Finished loading data, pos_id length %d, train answers length %d, test answers length %d", len(train_pos_id), len(train_answers), len(test_answers))

    return (passage_text, train_pos_id, train_answers, test_answers, test_answers_trivia, dev_answers, dev_answers_trivia)

from collections import OrderedDict
def load_model(args, checkpoint_path,load_flag=False):
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    configObj = MSMarcoConfigDict[args.model_type]
    args.model_name_or_path = checkpoint_path

    model = configObj.model_class(args)

    if args.init_from_fp16_ckpt:
        checkpoint_step = checkpoint_path.split('-')[-1].replace('/','')
        init_step = args.pretrained_checkpoint_dir.split('-')[-1].replace('/','')
        load_flag = checkpoint_step > init_step

    if args.fp16 and load_flag:
        checkpoint = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        saved_state = load_states_from_checkpoint(checkpoint_path)
        model_to_load = get_model_obj(model)
        logger.info('Loading saved model state ...')
        model_to_load.load_state_dict(saved_state.model_dict)
    
    model.is_representation_l2_normalization = args.representation_l2_normalization
    
    model.to(args.device)
    logger.info("Inference parameters %s", args)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )
    return model


def InferenceEmbeddingFromStreamDataLoader(args, model, train_dataloader, is_query_inference = True, prefix =""):
    # expect dataset from ReconstructTrainingSet
    results = {}
    eval_batch_size = args.per_gpu_eval_batch_size

    # Inference!
    logger.info("***** Running ANN Embedding Inference *****")
    logger.info("  Batch size = %d", eval_batch_size)

    embedding = []
    embedding2id = []

    if args.local_rank != -1:
        dist.barrier()
    model.eval()

    if (args.emb_file_multi_split_num > 0) and ("passage" in prefix):
        if args.emb_file_multi_split_size <=0 :
            logger.error(f"not specificying the sub split size of the emb files for sub preocess rank {args.rank}")
            exit()
        sub_splitting_size = args.emb_file_multi_split_size
        sub_split_count = 0
    else:
        sub_splitting_size = -1

    for batch in tqdm(train_dataloader, desc="Inferencing", disable=args.local_rank not in [-1, 0], position=0, leave=True):
        idxs = batch[3].detach().numpy() #[#B]

        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0].long(), "attention_mask": batch[1].long()}
            if is_query_inference:
                embs = model.module.query_emb(**inputs)
            else:
                embs = model.module.body_emb(**inputs)

        embs = embs.detach().cpu().numpy()

        # check for multi chunk output for long sequence 
        if len(embs.shape) == 3:
            for chunk_no in range(embs.shape[1]):
                embedding2id.append(idxs)
                embedding.append(embs[:,chunk_no,:])
        else:
            embedding2id.append(idxs)
            embedding.append(embs)

        if (sub_splitting_size > 0) and (len(embedding2id)*eval_batch_size >= sub_splitting_size):
            assert sub_split_count + 1 <= args.emb_file_multi_split_num
            embedding = np.concatenate(embedding, axis=0)
            embedding2id = np.concatenate(embedding2id, axis=0)
            
            emb_file_rank_id = args.rank * args.emb_file_multi_split_num + sub_split_count
            pickle_write(args=args,rank_id=emb_file_rank_id, prefix=prefix+"_emb_p_", data_array=embedding)
            pickle_write(args=args,rank_id=emb_file_rank_id, prefix=prefix+"_embid_p_", data_array=embedding2id)
            del embedding
            del embedding2id
            embedding = []
            embedding2id = []           
            sub_split_count = sub_split_count + 1

    # handle for remainders 
    if  (sub_splitting_size > 0 ) and (len(embedding2id) > 0):
        assert sub_split_count + 1 <= args.emb_file_multi_split_num
        embedding = np.concatenate(embedding, axis=0)
        embedding2id = np.concatenate(embedding2id, axis=0)        
        emb_file_rank_id = args.rank * args.emb_file_multi_split_num + sub_split_count
        pickle_write(args=args,rank_id=emb_file_rank_id, prefix=prefix+"_emb_p_", data_array=embedding)
        pickle_write(args=args,rank_id=emb_file_rank_id, prefix=prefix+"_embid_p_", data_array=embedding2id)
        del embedding
        del embedding2id
        sub_split_count = sub_split_count + 1
        # logger.info(f"process rank {args.rank} save {sub_split_count} splitted emb files to disk")
        logger.warning(
            "Process rank: %s, save %s splitted emb files to disk",
            args.local_rank,
            sub_split_count
        )
        return None,None
    else:
        embedding = np.concatenate(embedding, axis=0)
        embedding2id = np.concatenate(embedding2id, axis=0)
        return embedding, embedding2id


# streaming inference
def StreamInferenceDoc(args, model, fn, prefix, f, is_query_inference = True, load_cache=False):
    inference_batch_size = args.per_gpu_eval_batch_size #* max(1, args.n_gpu)
    #inference_dataloader = StreamingDataLoader(f, fn, batch_size=inference_batch_size, num_workers=1)
    inference_dataset = StreamingDataset(f, fn)
    inference_dataloader = DataLoader(inference_dataset, batch_size=inference_batch_size)

    if args.local_rank != -1:
        dist.barrier() # directory created

    if (args.emb_file_multi_split_num > 0) and ("passage" in prefix):
        # extra handling the memory problem by specifying the size of file
        _, _ = InferenceEmbeddingFromStreamDataLoader(args, model, inference_dataloader, is_query_inference = is_query_inference, prefix = prefix)
        # dist.barrier()
        full_embedding = None
        full_embedding2id = None # TODO: loading ids for first_worker()
    else:
        if load_cache:
            _embedding = None
            _embedding2id = None
        else:
            _embedding, _embedding2id = InferenceEmbeddingFromStreamDataLoader(args, model, inference_dataloader, is_query_inference = is_query_inference, prefix = prefix)

        not_loading = args.split_ann_search and ("passage" in prefix)
        # preserve to memory
        full_embedding = barrier_array_merge(args, _embedding, prefix = prefix + "_emb_p_", load_cache = load_cache, only_load_in_master = True,not_loading=not_loading)
        _embedding=None
        del _embedding
        full_embedding2id = barrier_array_merge(args, _embedding2id, prefix = prefix + "_embid_p_", load_cache = load_cache, only_load_in_master = True,not_loading=not_loading)
        logger.info( f"finish saving embbedding of {prefix}, not loading into MEM: {not_loading}" )
        _embedding2id=None
        del  _embedding2id

    return full_embedding, full_embedding2id

def inference_or_load_embedding(args,logger,model,checkpoint_path,text_data_prefix, emb_prefix, is_query_inference=True,checkonly=False,load_emb=True):
    # logging.info(f"checkpoint_path {checkpoint_path}")
    checkpoint_step = checkpoint_path.split('-')[-1].replace('/','')
    emb_file_pattern = os.path.join(args.output_dir,f'{emb_prefix}{checkpoint_step}__emb_p__data_obj_*.pb')
    emb_file_lists = glob.glob(emb_file_pattern)
    emb_file_lists = sorted(emb_file_lists, key=lambda name: int(name.split('_')[-1].replace('.pb',''))) # sort with split num
    logger.info(f"pattern {emb_file_pattern}\n file lists: {emb_file_lists}")
    embedding,embedding2id = [None,None]
    if len(emb_file_lists) > 0:
        if is_first_worker():
            logger.info(f"***** found existing embedding files {emb_file_pattern}, loading... *****")
            if checkonly:
                logger.info("check embedding files only, not loading")
                return embedding,embedding2id
            embedding = []
            embedding2id = []
            for emb_file in emb_file_lists:
                if load_emb:
                    with open(emb_file,'rb') as handle:
                        embedding.append(pickle.load(handle))
                embid_file = emb_file.replace('emb_p','embid_p')
                with open(embid_file,'rb') as handle:
                    embedding2id.append(pickle.load(handle))
            if (load_emb and not embedding) or (not embedding2id):
                logger.error("No data found for checkpoint: ",emb_file_pattern)
            if load_emb:
                embedding = np.concatenate(embedding, axis=0)
            embedding2id = np.concatenate(embedding2id, axis=0)
        # return embedding,embedding2id
    # else:
    #    if args.local_rank != -1:
    #        dist.barrier() # if multi-processing
    else:
        logger.info(f"***** inference of {text_data_prefix} *****")
        query_collection_path = os.path.join(args.data_dir, text_data_prefix)
        query_cache = EmbeddingCache(query_collection_path)
        with query_cache as emb:
            embedding,embedding2id = StreamInferenceDoc(args, model, 
                GetProcessingFn(args, query=is_query_inference),
                emb_prefix + str(checkpoint_step) + "_", emb,
                is_query_inference=is_query_inference)
    return embedding,embedding2id


def generate_new_ann(args, output_num, checkpoint_path, preloaded_data, latest_step_num):

    model = load_model(args, checkpoint_path)
    pid2offset, offset2pid = load_mapping(args.data_dir, "pid2offset")
    checkpoint_step = checkpoint_path.split('-')[-1].replace('/','')

    query_embedding, query_embedding2id = inference_or_load_embedding(args=args,logger=logger,model=model, checkpoint_path=checkpoint_path,text_data_prefix="train-query", emb_prefix="query_", is_query_inference=True)
    dev_query_embedding, dev_query_embedding2id = inference_or_load_embedding(args=args,logger=logger,model=model, checkpoint_path=checkpoint_path,text_data_prefix="test-query", emb_prefix="dev_query_", is_query_inference=True)
    dev_query_embedding_trivia, dev_query_embedding2id_trivia = inference_or_load_embedding(args=args,logger=logger,model=model, checkpoint_path=checkpoint_path,text_data_prefix="trivia-test-query", emb_prefix="trivia_dev_query_", is_query_inference=True)
    real_dev_query_embedding, real_dev_query_embedding2id = inference_or_load_embedding(args=args,logger=logger,model=model, checkpoint_path=checkpoint_path,text_data_prefix="dev-qas-query", emb_prefix="real-dev_query_", is_query_inference=True)
    real_dev_query_embedding_trivia, real_dev_query_embedding2id_trivia = inference_or_load_embedding(args=args,logger=logger,model=model, checkpoint_path=checkpoint_path,text_data_prefix="trivia-dev-qas-query", emb_prefix="trivia_real-dev_query_", is_query_inference=True)

    # passage_embedding == None, if args.split_ann_search == True
    passage_embedding, passage_embedding2id = inference_or_load_embedding(args=args,logger=logger,model=model, checkpoint_path=checkpoint_path,text_data_prefix="passages", emb_prefix="passage_", is_query_inference=False,load_emb= not args.split_ann_search)
    
    
    if args.gpu_index:
        del model  # leave gpu for faiss
        torch.cuda.empty_cache()
        time.sleep(10)

    if args.local_rank != -1:
        dist.barrier()

    # if None, reloading
    if passage_embedding2id is None and is_first_worker():
        _, passage_embedding2id = inference_or_load_embedding(args=args,logger=logger,model=None, checkpoint_path=checkpoint_path,text_data_prefix="passages", emb_prefix="passage_", is_query_inference=False,load_emb=False)
        logger.info(f"document id size: {passage_embedding2id.shape}")

    if is_first_worker():
        # passage_text, train_pos_id, train_answers, test_answers, test_answers_trivia = preloaded_data
        passage_text, train_pos_id, train_answers, test_answers, test_answers_trivia, dev_answers, dev_answers_trivia = preloaded_data

        if not args.split_ann_search:
            dim = passage_embedding.shape[1]
            print('passage embedding shape: ' + str(passage_embedding.shape))
            top_k = args.topk_training 
            faiss.omp_set_num_threads(16)
            cpu_index = faiss.IndexFlatIP(dim)
            index = get_gpu_index(cpu_index) if args.gpu_index else cpu_index
            index.add(passage_embedding)
            logger.info("***** Done ANN Index *****")
            _, dev_I = index.search(dev_query_embedding, 100) #I: [number of queries, topk]            
            _, dev_I_trivia = index.search(dev_query_embedding_trivia, 100) #I: [number of queries, topk]
            logger.info("Start searching for query embedding with length %d", len(query_embedding))
            _, I = index.search(query_embedding, top_k) #I: [number of queries, topk]
        else:
            _, dev_I_trivia, real_dev_D, real_dev_I  = document_split_faiss_index(
                    logger=logger,
                    args=args,
                    top_k_dev=100,
                    top_k=args.topk_training,
                    checkpoint_step=checkpoint_step,
                    dev_query_emb=dev_query_embedding_trivia,
                    train_query_emb=real_dev_query_embedding,
                    emb_prefix="passage_",two_query_set=True,
            )
            dev_D, dev_I, _, I  = document_split_faiss_index(
                    logger=logger,
                    args=args,
                    top_k_dev=100,
                    top_k=args.topk_training,
                    checkpoint_step=checkpoint_step,
                    dev_query_emb=dev_query_embedding,
                    train_query_emb=query_embedding,
                    emb_prefix="passage_",two_query_set=True,
            )
        save_trec_file(
            dev_query_embedding2id,passage_embedding2id,dev_I,dev_D,
            trec_save_path= os.path.join(os.path.join(args.training_dir,"ann_data", "nq-test_" + checkpoint_step + ".trec")),
            topN=100
        )
        save_trec_file(
            real_dev_query_embedding2id,passage_embedding2id,real_dev_I,real_dev_D,
            trec_save_path= os.path.join(os.path.join(args.training_dir,"ann_data", "nq-dev_" + checkpoint_step + ".trec")),
            topN=100
        )
        # measure ANN mrr 
        top_k_hits = validate(passage_text, test_answers, dev_I, dev_query_embedding2id, passage_embedding2id)
        real_dev_top_k_hits = validate(passage_text, dev_answers, real_dev_I, real_dev_query_embedding2id, passage_embedding2id)
        top_k_hits_trivia = validate(passage_text, test_answers_trivia, dev_I_trivia, dev_query_embedding2id_trivia, passage_embedding2id)
        query_range_number = I.shape[0]
        json_dump_dict = {
            'top20': top_k_hits[19], 'top100': top_k_hits[99], 'top20_trivia': top_k_hits_trivia[19],
            'dev_top20': real_dev_top_k_hits[19], 'dev_top100': real_dev_top_k_hits[99],  
            'top100_trivia': top_k_hits_trivia[99], 'checkpoint': checkpoint_path, 'n_train_query':query_range_number,
        }
        logger.info(json_dump_dict)

        
        logger.info("***** GenerateNegativePassaageID *****")
        effective_q_id = set(query_embedding2id.flatten())

        logger.info("Effective qid length %d, search result length %d", len(effective_q_id), I.shape[0])
        query_negative_passage = GenerateNegativePassaageID(args, passage_text, train_answers, query_embedding2id, passage_embedding2id, I, train_pos_id)

        logger.info("Done generating negative passages, output length %d", len(query_negative_passage))

        if args.dual_training:
            
            assert args.split_ann_search and args.gpu_index # hard set
            logger.info("***** Begin ANN Index for dual d2q task *****")
            top_k = args.topk_training
            faiss.omp_set_num_threads(args.faiss_omp_num_threads)
            logger.info("***** Faiss: total {} gpus *****".format(faiss.get_num_gpus()))
            cpu_index = faiss.IndexFlatIP(query_embedding.shape[1])
            index = get_gpu_index(cpu_index) if args.gpu_index else cpu_index
            index.add(query_embedding)
            logger.info("***** Done building ANN Index for dual d2q task *****")

            # train_pos_id : a list, idx -> int pid
            train_pos_id_inversed = {}
            for qidx in range(query_embedding2id.shape[0]):
                qid = query_embedding2id[qidx]
                pid = int(train_pos_id[qid])
                if pid not in train_pos_id_inversed:
                    train_pos_id_inversed[pid]=[qid]
                else:
                    train_pos_id_inversed[pid].append(qid)
            
            possitive_training_passage_id = [ train_pos_id[t] for t in query_embedding2id] # 
            # compatible with MaxP
            possitive_training_passage_id_embidx=[]
            possitive_training_passage_id_to_subset_embidx={} # pid to indexs in pos_pas_embs 
            possitive_training_passage_id_emb_counts=0
            for pos_pid in possitive_training_passage_id:
                embidx=np.asarray(np.where(passage_embedding2id==pos_pid)).flatten()
                possitive_training_passage_id_embidx.append(embidx)
                possitive_training_passage_id_to_subset_embidx[int(pos_pid)] = np.asarray(range(possitive_training_passage_id_emb_counts,possitive_training_passage_id_emb_counts+embidx.shape[0]))
                possitive_training_passage_id_emb_counts += embidx.shape[0]
            possitive_training_passage_id_embidx=np.concatenate(possitive_training_passage_id_embidx,axis=0)
            
            if not args.split_ann_search:
                D, I = index.search(passage_embedding[possitive_training_passage_id_embidx], args.topk_training_d2q) 
            else:
                positive_p_embs = loading_possitive_document_embedding(logger,args.output_dir,checkpoint_step,possitive_training_passage_id_embidx,emb_prefix="passage_",)
                assert positive_p_embs.shape[0] == len(possitive_training_passage_id)
                D, I = index.search(positive_p_embs, args.topk_training_d2q) 
                positive_p_embs = None
                del positive_p_embs
            index.reset()
            logger.info("***** Finish ANN searching for dual d2q task, construct  *****")
            passage_negative_queries = GenerateNegativeQueryID(args, passage_text,train_answers, query_embedding2id, passage_embedding2id[possitive_training_passage_id_embidx], closest_ans=I, training_query_positive_id_inversed=train_pos_id_inversed)
            logger.info("***** Done ANN searching for negative queries *****")


        logger.info("***** Construct ANN Triplet *****")
        prefix =  "ann_grouped_training_data_" if args.grouping_ann_data > 0  else "ann_training_data_"
        train_data_output_path = os.path.join(
            args.output_dir, prefix + str(output_num))
        query_range = list(range(query_range_number))
        random.shuffle(query_range)
        if args.grouping_ann_data > 0 :
            with open(train_data_output_path, 'w') as f:
                counting=0
                pos_q_group={}
                pos_d_group={}
                neg_D_group={} # {0:[], 1:[], 2:[]...}
                if args.dual_training:
                    neg_Q_group={}
                for query_idx in query_range: 
                    query_id = query_embedding2id[query_idx]
                    pos_pid = train_pos_id[query_id]
                    
                    pos_q_group[counting]=int(query_id)
                    pos_d_group[counting]=int(pos_pid)

                    neg_D_group[counting]=[int(neg_pid) for neg_pid in query_negative_passage[query_id]]
                    if args.dual_training:
                        neg_Q_group[counting]=[int(neg_qid) for neg_qid in passage_negative_queries[pos_pid]]
                    counting +=1
                    if counting >= args.grouping_ann_data:
                        jsonline_dict={}
                        jsonline_dict["pos_q_group"]=pos_q_group
                        jsonline_dict["pos_d_group"]=pos_d_group
                        jsonline_dict["neg_D_group"]=neg_D_group
                        
                        if args.dual_training:
                            jsonline_dict["neg_Q_group"]=neg_Q_group

                        f.write(f"{json.dumps(jsonline_dict)}\n")

                        counting=0
                        pos_q_group={}
                        pos_d_group={}
                        neg_D_group={} # {0:[], 1:[], 2:[]...}
                        if args.dual_training:
                            neg_Q_group={}
             
        else:
            # not support dualtraining
            with open(train_data_output_path, 'w') as f:
                for query_idx in query_range: 
                    query_id = query_embedding2id[query_idx]
                    # if not query_id in train_pos_id:
                    #     continue
                    pos_pid = train_pos_id[query_id]
                    
                    if not args.dual_training:
                        f.write(
                            "{}\t{}\t{}\n".format(
                                query_id, pos_pid,
                                ','.join(
                                    str(neg_pid) for neg_pid in query_negative_passage[query_id])))
                    else:
                        # if pos_pid not in effective_p_id or pos_pid not in training_query_positive_id_inversed:
                        #     continue
                        f.write(
                            "{}\t{}\t{}\t{}\n".format(
                                query_id, pos_pid,
                                ','.join(
                                    str(neg_pid) for neg_pid in query_negative_passage[query_id]),
                                ','.join(
                                    str(neg_qid) for neg_qid in passage_negative_queries[pos_pid])
                            )
                        )
        ndcg_output_path = os.path.join(args.output_dir, "ann_ndcg_" + str(output_num))
        with open(ndcg_output_path, 'w') as f:
            json.dump(json_dump_dict, f)


def GenerateNegativePassaageID(args, passages, answers, query_embedding2id, passage_embedding2id, closest_docs, training_query_positive_id):
    query_negative_passage = {}

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    for query_idx in range(closest_docs.shape[0]): 
        query_id = query_embedding2id[query_idx]

        pos_pid = training_query_positive_id[query_id]
        doc_ids = [passage_embedding2id[pidx] for pidx in closest_docs[query_idx]]

        query_negative_passage[query_id] = []
        neg_cnt = 0

        for doc_id in doc_ids:
            if doc_id == pos_pid:
                continue
            if doc_id in query_negative_passage[query_id]:
                continue
            if neg_cnt >= args.negative_sample:
                break
            
            text = passages[doc_id][0]
            if not has_answer(answers[query_id], text, tokenizer):
                query_negative_passage[query_id].append(doc_id)
            neg_cnt+=1 # BUG?

    return query_negative_passage

# TODO: change to DPR type
def GenerateNegativeQueryID(args, passages, answers, query_embedding2id, passage_embedding2id, closest_ans, training_query_positive_id_inversed):
    passage_negative_query = {}

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    for passage_idx in range(closest_ans.shape[0]):
        passage_id = passage_embedding2id[passage_idx]

        pos_qid_list = training_query_positive_id_inversed[passage_id]
        q_ids = [query_embedding2id[qidx] for qidx in closest_ans[passage_idx] ]

        passage_negative_query[passage_id] = []
        neg_cnt = 0

        text = passages[passage_id][0]

        for q_id in q_ids:
            if q_id in pos_qid_list:
                continue
            if q_id in passage_negative_query[passage_id]:
                continue
            if neg_cnt >= args.negative_sample:
                break
            if not has_answer(answers[q_id],text,tokenizer):
                passage_negative_query[passage_id].append(q_id)
            neg_cnt+=1

    return passage_negative_query


def validate(passages, answers, closest_docs, query_embedding2id, passage_embedding2id):

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    logger.info('Matching answers in top docs...')
    scores = []
    for query_idx in range(closest_docs.shape[0]): 
        query_id = query_embedding2id[query_idx]
        doc_ids = [passage_embedding2id[pidx] for pidx in closest_docs[query_idx]]
        hits = []
        for i, doc_id in enumerate(doc_ids):
            text = passages[doc_id][0]
            hits.append(has_answer(answers[query_id], text, tokenizer))
        scores.append(hits)

    logger.info('Per question validation results len=%d', len(scores))

    n_docs = len(closest_docs[0])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(closest_docs) for v in top_k_hits]
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return top_k_hits
    

def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--training_dir",
        default=None,
        type=str,
        required=True,
        help="Training dir, will look for latest checkpoint dir in here",
    )
    parser.add_argument(
        "--init_model_dir",
        default=None,
        type=str,
        required=True,
        help="Initial model dir, will use this if no checkpoint is found in model_dir",
    )
    parser.add_argument(
        "--last_checkpoint_dir",
        default="",
        type=str,
        help="Last checkpoint used, this is for rerunning this script when some ann data is already generated",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MSMarcoConfigDict.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the training data will be written",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        required=True,
        help="The directory where cached data will be written",
    )
    parser.add_argument(
        "--end_output_num",
        default=-1,
        type=int,
        help="Stop after this number of data versions has been generated, default run forever",
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

    parser.add_argument(
        "--max_doc_character",
        default= 10000, 
        type=int,
        help="used before tokenizer to save tokenizer latency",
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=128,
        type=int,
        help="The starting output file number",
    )

    parser.add_argument(
        "--ann_chunk_factor",
        default= 5, # for 500k queryes, divided into 100k chunks for each epoch
        type=int,
        help="devide training queries into chunks",
    )

    parser.add_argument(
        "--topk_training",
        default= 500,
        type=int,
        help="top k from which negative samples are collected",
    )

    parser.add_argument(
        "--negative_sample",
        default= 5,
        type=int,
        help="at each resample, how many negative samples per query do I use",
    )

    parser.add_argument(
        "--topk_training_d2q",
        default=200,
        type=int,
        help="top k from which negative samples are collected",
    )

    parser.add_argument(
        "--ann_measure_topk_mrr",
        default = False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--only_keep_latest_embedding_file",
        default = False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument(
        "--passage_path",
        default=None,
        type=str,
        required=True,
        help="passage_path",
    )

    parser.add_argument(
        "--test_qa_path",
        default=None,
        type=str,
        required=True,
        help="test_qa_path",
    )

    parser.add_argument(
        "--trivia_test_qa_path",
        default=None,
        type=str,
        required=True,
        help="trivia_test_qa_path",
    )
    
    parser.add_argument(
        "--dual_training",
        action="store_true",
        help="enable dual training, change the data loading, forward function and loss function",
    )
    parser.add_argument(
        "--faiss_omp_num_threads", 
        type=int, 
        default=16, 
        help="for faiss.omp_set_num_threads()",
    )
    parser.add_argument(
        "--split_ann_search",
        default=False,
        action="store_true",
        help="separately do ANN index and merge result",
    )
    parser.add_argument(
        "--gpu_index",
        default=False,
        action="store_true",
        help="separately do ANN index and merge result",
    )

    parser.add_argument(
        "--emb_file_multi_split_num",
        default= -1,
        type=int,
        help="extra splitting of the embeddings",
    )
    parser.add_argument(
        "--emb_file_multi_split_size",
        default= -1,
        type=int,
        help="extra splitting of the embeddings max size",
    )
    parser.add_argument(
        "--grouping_ann_data",
        type=int, 
        default=-1,         
        help="group multiple <q,d> pair data into one line, I prefer set to 32",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--init_from_fp16_ckpt",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--representation_l2_normalization",
        action="store_true",
        help="enable l2_normalization on the representative embeddings for ANN retrieval, previously named as --l2_normalization",
    )

    args = parser.parse_args()

    return args


def set_env(args):
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

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Calculate split size
    if (args.emb_file_multi_split_num > 0):
        with open(os.path.join(args.data_dir,"passages_meta"),"r") as f:
            n_passage_overall = json.load(f)['total_number']
        n_passage_per_gpu = int( n_passage_overall / args.world_size ) + 1
        args.emb_file_multi_split_size = int( n_passage_per_gpu / args.emb_file_multi_split_num ) + 1

def ann_data_gen(args):
    last_checkpoint = args.last_checkpoint_dir
    ann_no, ann_path, ndcg_json = get_latest_ann_data(args.output_dir)
    output_num = ann_no + 1

    logger.info("starting output number %d", output_num)
    preloaded_data = None
    
    if is_first_worker():
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)
        preloaded_data = load_data(args)

    while args.end_output_num == -1 or output_num <= args.end_output_num:
        next_checkpoint, latest_step_num = get_latest_checkpoint(args)
        logger.info(f"get next_checkpoint {next_checkpoint} latest_step_num {latest_step_num} ")
        if args.only_keep_latest_embedding_file:
            latest_step_num = 0

        if next_checkpoint == last_checkpoint:
            time.sleep(60)
        else:
            logger.info("start generate ann data number %d", output_num)
            logger.info("next checkpoint at " + next_checkpoint)
            generate_new_ann(args, output_num, next_checkpoint, preloaded_data, latest_step_num)
            logger.warning("process rank: %s, finished generating ann data number %d", args.local_rank, output_num)
            # logger.info("finished generating ann data number %d", output_num)
            output_num += 1
            last_checkpoint = next_checkpoint
        if args.local_rank != -1:
            dist.barrier()


def main():
    args = get_arguments()
    set_env(args)
    ann_data_gen(args)


if __name__ == "__main__":
    main()