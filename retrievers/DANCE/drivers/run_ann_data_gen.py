import sys
sys.path += ['../']
import torch
import os
import faiss
from utils.util import (
    barrier_array_merge,
    convert_to_string_id,
    is_first_worker,
    StreamingDataset,
    EmbeddingCache,
    get_checkpoint_no,
    get_latest_ann_data
)
import csv
import copy
import transformers
from transformers import (
    AdamW,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    RobertaModel,
)
from data.msmarco_data import GetProcessingFn  
from model.models import MSMarcoConfigDict, ALL_MODELS 
from model.models import replace_model_with_spasre
from torch import nn
import torch.distributed as dist
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import numpy as np
from os.path import isfile, join
import argparse
import json
import logging
import random
import time
import pytrec_eval
import glob
import pickle

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score

from utils.metric import Metric
from utils.trec_convert import save_trec_file,convert_trec_to_MARCO_id
from utils.indexing_utils import clean_faiss_gpu,get_gpu_index,document_split_faiss_index,loading_possitive_document_embedding

torch.multiprocessing.set_sharing_strategy('file_system')


logger = logging.getLogger(__name__)


# ANN - active learning ------------------------------------------------------

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


def get_latest_checkpoint(args):
    if not os.path.exists(args.training_dir):
        return args.init_model_dir, 0
    subdirectories = list(next(os.walk(args.training_dir))[1])

    def valid_checkpoint(checkpoint):
        chk_path = os.path.join(args.training_dir, checkpoint)
        scheduler_path = os.path.join(chk_path, "scheduler.pt")
        return os.path.exists(scheduler_path)

    checkpoint_nums = [get_checkpoint_no(
        s) for s in subdirectories if valid_checkpoint(s)]

    if len(checkpoint_nums) > 0:
        return os.path.join(args.training_dir, "checkpoint-" +
                            str(max(checkpoint_nums))) + "/", max(checkpoint_nums)
    return args.init_model_dir, 0


def load_positive_ids(args):

    logger.info("Loading query_2_pos_docid")
    training_query_positive_id = {}
    query_positive_id_path = os.path.join(args.data_dir, "train-qrel.tsv")
    with open(query_positive_id_path, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, docid, rel] in tsvreader:
            assert rel == "1"
            topicid = int(topicid)
            docid = int(docid)
            training_query_positive_id[topicid] = docid

    logger.info("Loading dev query_2_pos_docid")
    dev_query_positive_id = {}

    query_positive_id_path = os.path.join(args.data_dir, "real-dev-qrel.tsv")

    with open(query_positive_id_path, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, docid, rel] in tsvreader:
            topicid = int(topicid)
            docid = int(docid)
            if topicid not in dev_query_positive_id:
                dev_query_positive_id[topicid] = {}
            dev_query_positive_id[topicid][docid] = int(rel)

    return training_query_positive_id, dev_query_positive_id


def load_model(args, checkpoint_path):
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    configObj = MSMarcoConfigDict[args.model_type]
    args.model_name_or_path = checkpoint_path
    config = configObj.config_class.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="MSMarco",
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = configObj.tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = configObj.model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # enable l2 normalization on representation H layer
    model.is_representation_l2_normalization = args.representation_l2_normalization

    model.to(args.device)
    logger.info("Inference parameters %s", args)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    return config, tokenizer, model


def InferenceEmbeddingFromStreamDataLoader(
        args,
        model,
        train_dataloader,
        is_query_inference=True,
        prefix=""):
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

    # logger_count=0
    # for batch in train_dataloader:
    #     logger_count = logger_count + 1
    #     if logger_count % 1000 == 0:
    #         t= time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #         logger.info(f"Inferencing batch {logger_count}, current time:{t}")
    for batch in tqdm(train_dataloader,
                      desc="Inferencing",
                      disable=args.local_rank not in [-1,
                                                      0],
                      position=0,
                      leave=True):

        idxs = batch[3].detach().numpy()  # [#B]

        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1].long()}
            if is_query_inference:
                embs = model.module.query_emb(**inputs)
            else:
                embs = model.module.body_emb(**inputs)

        embs = embs.detach().cpu().numpy()

        # check for multi chunk output for long sequence
        if len(embs.shape) == 3:
            for chunk_no in range(embs.shape[1]):
                embedding2id.append(idxs)
                embedding.append(embs[:, chunk_no, :])
        else:
            embedding2id.append(idxs)
            embedding.append(embs)

    embedding = np.concatenate(embedding, axis=0)
    embedding2id = np.concatenate(embedding2id, axis=0)
    return embedding, embedding2id


# streaming inference
def StreamInferenceDoc(args, model, fn, prefix, f, is_query_inference=True):
    inference_batch_size = args.per_gpu_eval_batch_size  # * max(1, args.n_gpu)
    inference_dataset = StreamingDataset(f, fn)
    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=inference_batch_size)

    if args.local_rank != -1:
        dist.barrier()  # directory created

    _embedding, _embedding2id = InferenceEmbeddingFromStreamDataLoader(
        args, model, inference_dataloader, is_query_inference=is_query_inference, prefix=prefix)

    logger.info("merging embeddings")

    not_loading = args.split_ann_search and ("passage" in prefix)

    # preserve to memory
    full_embedding = barrier_array_merge(
        args,
        _embedding,
        prefix=prefix +
        "_emb_p_",
        load_cache=False,
        only_load_in_master=True,
        not_loading=not_loading)
    _embedding=None
    del _embedding
    logger.info( f"finish saving embbedding of {prefix}, not loading into MEM: {not_loading}" )

    full_embedding2id = barrier_array_merge(
        args,
        _embedding2id,
        prefix=prefix +
        "_embid_p_",
        load_cache=False,
        only_load_in_master=True)

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
        if args.local_rank != -1:
            dist.barrier() # if multi-processing
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

def generate_new_ann(
        args,
        output_num,
        checkpoint_path,
        training_query_positive_id,
        dev_query_positive_id,
        latest_step_num):
    if args.gpu_index:
        clean_faiss_gpu()
    if not args.not_load_model_for_inference:
        config, tokenizer, model = load_model(args, checkpoint_path)
    
    checkpoint_step = checkpoint_path.split('-')[-1].replace('/','')

    def evaluation(dev_query_embedding2id,passage_embedding2id,dev_I,dev_D,trec_prefix="real-dev_query_",test_set="trec2019",split_idx=-1,d2q_eval=False,d2q_qrels=None):
        if d2q_eval:
            qrels=d2q_qrels
        else:
            if args.data_type ==0 :
                if not d2q_eval:
                    if  test_set== "marcodev":
                        qrels="../data/raw_data/msmarco-docdev-qrels.tsv"
                    elif test_set== "trec2019":
                        qrels="../data/raw_data/2019qrels-docs.txt"
            elif args.data_type ==1:
                if test_set == "marcodev":
                    qrels="../data/raw_data/qrels.dev.small.tsv"
            else:
                logging.error("wrong data type")
                exit()
        trec_path=os.path.join(args.output_dir, trec_prefix + str(checkpoint_step)+".trec")
        save_trec_file(
            dev_query_embedding2id,passage_embedding2id,dev_I,dev_D,
            trec_save_path= trec_path,
            topN=200)
        convert_trec_to_MARCO_id(
            data_type=args.data_type,test_set=test_set,
            processed_data_dir=args.data_dir,
            trec_path=trec_path,d2q_reversed_trec_file=d2q_eval)

        trec_path=trec_path.replace(".trec",".formatted.trec")
        met = Metric()
        if split_idx >= 0:
            split_file_path=qrels+f"{args.dev_split_num}_fold.split_dict"
            with open(split_file_path,'rb') as f:
                split=pickle.load(f)
        else:
            split=None
        
        ndcg10 = met.get_metric(qrels, trec_path, 'ndcg_cut_10',split,split_idx)
        mrr10 = met.get_mrr(qrels, trec_path, 'mrr_cut_10',split,split_idx)
        mrr100 = met.get_mrr(qrels, trec_path, 'mrr_cut_100',split,split_idx)

        logging.info(f" evaluation for {test_set}, trec_file {trec_path}, split_idx {split_idx} \
            ndcg_cut_10 : {ndcg10}, \
            mrr_cut_10 : {mrr10}, \
            mrr_cut_100 : {mrr100}"
        )

        return ndcg10


    # Inference 
    if args.data_type==0:
        # TREC DL 2019 evalset
        trec2019_query_embedding, trec2019_query_embedding2id = inference_or_load_embedding(args=args,logger=logger,model=model, checkpoint_path=checkpoint_path,text_data_prefix="dev-query", emb_prefix="dev_query_", is_query_inference=True)# it's trec-dl testset actually
    dev_query_embedding, dev_query_embedding2id = inference_or_load_embedding(args=args,logger=logger,model=model, checkpoint_path=checkpoint_path, text_data_prefix="real-dev-query", emb_prefix="real-dev_query_", is_query_inference=True)
    query_embedding, query_embedding2id = inference_or_load_embedding(args=args,logger=logger,model=model, checkpoint_path=checkpoint_path,text_data_prefix="train-query", emb_prefix="query_", is_query_inference=True)
    if not args.split_ann_search:
        # merge all passage
        passage_embedding, passage_embedding2id = inference_or_load_embedding(args=args,logger=logger,model=model, checkpoint_path=checkpoint_path,text_data_prefix="passages", emb_prefix="passage_", is_query_inference=False)
    else:
        # keep id only
        _, passage_embedding2id = inference_or_load_embedding(args=args,logger=logger,model=model, checkpoint_path=checkpoint_path,text_data_prefix="passages", emb_prefix="passage_", is_query_inference=False,load_emb=False)
        
    # FirstP shape,
    # passage_embedding: [[vec_0], [vec_1], [vec_2], [vec_3] ...], 
    # passage_embedding2id: [id0, id1, id2, id3, ...]

    # MaxP shape,
    # passage_embedding: [[vec_0_0], [vec_0_1],[vec_0_2],[vec_0_3],[vec_1_0],[vec_1_1] ...], 
    # passage_embedding2id: [id0, id0, id0, id0, id1, id1 ...]
    if args.gpu_index:
        del model  # leave gpu for faiss
        torch.cuda.empty_cache()
        time.sleep(10)

    if args.inference:
        return

    if is_first_worker():
        # Construct new traing subset
        chunk_factor = args.ann_chunk_factor
        effective_idx = output_num % chunk_factor

        if chunk_factor <= 0:
            chunk_factor = 1
        num_queries = len(query_embedding)
        queries_per_chunk = num_queries // chunk_factor
        q_start_idx = queries_per_chunk * effective_idx
        q_end_idx = num_queries if (
            effective_idx == (
                chunk_factor -
                1)) else (
            q_start_idx +
            queries_per_chunk)
        logger.info(
            "Chunked {} query from {}".format(
                len(query_embedding[q_start_idx:q_end_idx]),
                num_queries))

        if not args.split_ann_search:
            dim = passage_embedding.shape[1]
            print('passage embedding shape: ' + str(passage_embedding.shape))
            top_k = args.topk_training
            faiss.omp_set_num_threads(args.faiss_omp_num_threads)
            cpu_index = faiss.IndexFlatIP(dim)
            logger.info("***** Faiss: total {} gpus *****".format(faiss.get_num_gpus()))
            index = get_gpu_index(cpu_index) if args.gpu_index else cpu_index
            index.add(passage_embedding)
            # for measure ANN mrr
            logger.info("search dev query")
            dev_D, dev_I = index.search(dev_query_embedding, 100) # I: [number of queries, topk]
            logger.info("finish")
            logger.info("search train query")
            D, I = index.search(query_embedding[q_start_idx:q_end_idx], top_k) # I: [number of queries, topk]
            logger.info("finish")
            index.reset()
        else:
            if args.data_type==0:
                trec2019_D, trec2019_I, _, _ = document_split_faiss_index(
                    logger=logger,
                    args=args,
                    checkpoint_step=checkpoint_step,
                    top_k_dev = 200,
                    top_k = args.topk_training,
                    dev_query_emb=trec2019_query_embedding,
                    train_query_emb=None,
                    emb_prefix="passage_",two_query_set=False,
                )
            dev_D, dev_I, D, I = document_split_faiss_index(
                logger=logger,
                args=args,
                checkpoint_step=checkpoint_step,
                top_k_dev = 200,
                top_k = args.topk_training,
                dev_query_emb=dev_query_embedding,
                train_query_emb=query_embedding[q_start_idx:q_end_idx],
                emb_prefix="passage_")
            logger.info("***** seperately process indexing *****")
        
        
        logger.info("***** Done ANN Index *****")

        # dev_ndcg, num_queries_dev = EvalDevQuery(
        #     args, dev_query_embedding2id, passage_embedding2id, dev_query_positive_id, dev_I)
        logger.info("***** Begin evaluation *****")
        eval_dict_todump={'checkpoint': checkpoint_path}

        if args.data_type==0:
            trec2019_ndcg = evaluation(trec2019_query_embedding2id,passage_embedding2id,trec2019_I,trec2019_D,trec_prefix="dev_query_",test_set="trec2019")
        if args.dev_split_num > 0:
            marcodev_ndcg = 0.0
            for i in range(args.dev_split_num):
                ndcg_10_dev_split_i = evaluation(dev_query_embedding2id,passage_embedding2id,dev_I,dev_D,trec_prefix="real-dev_query_",test_set="marcodev",split_idx=i)
                if i != args.testing_split_idx:
                    marcodev_ndcg += ndcg_10_dev_split_i
                
                eval_dict_todump[f'marcodev_split_{i}_ndcg_cut_10'] = ndcg_10_dev_split_i

            logger.info(f"average marco dev { marcodev_ndcg /(args.dev_split_num -1)}")
        else:
            marcodev_ndcg = evaluation(dev_query_embedding2id,passage_embedding2id,dev_I,dev_D,trec_prefix="real-dev_query_",test_set="marcodev",split_idx=-1)
        
        eval_dict_todump['marcodev_ndcg']=marcodev_ndcg

        query_range_number = I.shape[0]
        if args.save_training_query_trec:
            logger.info("***** Save the ANN searching for negative passages in trec file format *****")    
            trec_output_path=os.path.join(args.output_dir, "ann_training_query_retrieval_" + str(output_num)+".trec")
            save_trec_file(query_embedding2id[q_start_idx:q_end_idx],passage_embedding2id,I,D,trec_output_path,topN=args.topk_training)

        effective_q_id = set(query_embedding2id[q_start_idx:q_end_idx].flatten())
        query_negative_passage = GenerateNegativePassaageID(
            args,
            query_embedding2id[q_start_idx:q_end_idx],
            passage_embedding2id,
            training_query_positive_id,
            I,
            effective_q_id)
        logger.info("***** Done ANN searching for negative passages *****")

        if args.d2q_task_evaluation and args.d2q_task_marco_dev_qrels is not None:
            with open(os.path.join(args.data_dir,'pid2offset.pickle'),'rb') as f:
                pid2offset = pickle.load(f)
            real_dev_ANCE_ids=[]
            with open(args.d2q_task_marco_dev_qrels+f"{args.dev_split_num}_fold.split_dict","rb") as f:
                dev_d2q_split_dict=pickle.load(f)
            for i in dev_d2q_split_dict:
                for stringdocid in dev_d2q_split_dict[i]:
                    if args.data_type==0:
                        real_dev_ANCE_ids.append(pid2offset[int(stringdocid[1:])])
                    else:
                        real_dev_ANCE_ids.append(pid2offset[int(stringdocid)])
            real_dev_ANCE_ids = np.array(real_dev_ANCE_ids).flatten()
            real_dev_possitive_training_passage_id_embidx=[]
            for dev_pos_pid in real_dev_ANCE_ids:
                embidx=np.asarray(np.where(passage_embedding2id==dev_pos_pid)).flatten()
                real_dev_possitive_training_passage_id_embidx.append(embidx)
                # possitive_training_passage_id_to_subset_embidx[int(dev_pos_pid)] = np.asarray(range(possitive_training_passage_id_emb_counts,possitive_training_passage_id_emb_counts+embidx.shape[0]))
                # possitive_training_passage_id_emb_counts += embidx.shape[0]
            real_dev_possitive_training_passage_id_embidx=np.concatenate(real_dev_possitive_training_passage_id_embidx,axis=0)
            del pid2offset
            if not args.split_ann_search:
                real_dev_positive_p_embs = passage_embedding[real_dev_possitive_training_passage_id_embidx]
            else:
                real_dev_positive_p_embs = loading_possitive_document_embedding(logger,args.output_dir,checkpoint_step,real_dev_possitive_training_passage_id_embidx,emb_prefix="passage_",)
            logger.info("***** d2q task evaluation *****")
            cpu_index = faiss.IndexFlatIP(dev_query_embedding.shape[1])
            index = cpu_index
            # index = get_gpu_index(cpu_index) if args.gpu_index else cpu_index
            index.add(dev_query_embedding)
            real_dev_d2q_D, real_dev_d2q_I = index.search(real_dev_positive_p_embs, 200) 
            if args.dev_split_num > 0:
                d2q_marcodev_ndcg = 0.0
                for i in range(args.dev_split_num):
                    d2q_ndcg_10_dev_split_i = evaluation( 
                        real_dev_ANCE_ids,dev_query_embedding2id ,real_dev_d2q_I,real_dev_d2q_D,
                        trec_prefix="d2q-dual-task_real-dev_query_",test_set="marcodev",split_idx=i,d2q_eval=True,d2q_qrels=args.d2q_task_marco_dev_qrels)
                    if i != args.testing_split_idx:
                        d2q_marcodev_ndcg += d2q_ndcg_10_dev_split_i
                    eval_dict_todump[f'd2q_marcodev_split_{i}_ndcg_cut_10'] = ndcg_10_dev_split_i
                logger.info(f"average marco dev d2q task { d2q_marcodev_ndcg /(args.dev_split_num -1)}")
            else:
                d2q_marcodev_ndcg = evaluation(real_dev_ANCE_ids,dev_query_embedding2id ,real_dev_d2q_I,real_dev_d2q_D,
                    trec_prefix="d2q-dual-task_real-dev_query_",test_set="marcodev",split_idx=-1,d2q_eval=True,d2q_qrels=args.d2q_task_marco_dev_qrels)
            
            eval_dict_todump['d2q_marcodev_ndcg'] = d2q_marcodev_ndcg

        if args.dual_training:
            # do this before completely truncating the query embedding 
            logger.info("***** Do ANN Index for dual d2q task *****")
            top_k = args.topk_training
            faiss.omp_set_num_threads(args.faiss_omp_num_threads)
            logger.info("***** Faiss: total {} gpus *****".format(faiss.get_num_gpus()))
            cpu_index = faiss.IndexFlatIP(query_embedding.shape[1])
            index = get_gpu_index(cpu_index) if args.gpu_index else cpu_index
            index.add(query_embedding)
            logger.info("***** Done building ANN Index for dual d2q task *****")
            

            logger.info("***** use ANCE id to construct positive passage embedding index *****")
            training_query_positive_id_inversed = {} # {v:k for k,v in training_query_positive_id.items()} # doc_id : query_id
            for k in training_query_positive_id:
                pos_pid=training_query_positive_id[k]
                if pos_pid not in training_query_positive_id_inversed:
                    training_query_positive_id_inversed[pos_pid]=[k]
                else:
                    training_query_positive_id_inversed[pos_pid].append(k)
            
            possitive_training_passage_id = [ training_query_positive_id[t] for t in query_embedding2id[q_start_idx:q_end_idx]] # 
            effective_p_id = set(possitive_training_passage_id)

            if "BM25_retrieval" == args.query_likelihood_strategy:
                passage_negative_queries = {}
                logger.info("***** loading negative queries from BM25 search result *****")
                with open(args.bm25_top_d2q_path,"r") as f:
                    for line in f:
                        pid,qid,rank = line.strip().split("\t")
                        pid = int(pid)
                        qid = int(qid)
                        if (pid in effective_p_id) and (qid not in training_query_positive_id_inversed[pid]):
                            if pid not in passage_negative_queries:
                                passage_negative_queries[pid]=[qid]
                            elif len(passage_negative_queries[pid]) < args.topk_training_d2q:
                                passage_negative_queries[pid].append(qid)
                                
                logger.info(f"***** shuffle and pick {args.negative_sample} negative queries *****")
                for pid in passage_negative_queries:
                    random.shuffle(passage_negative_queries[pid])
                    passage_negative_queries[pid]=passage_negative_queries[pid][:args.negative_sample]

            else:
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
                    if args.query_likelihood_strategy == "positive_doc":
                        positive_p_embs = loading_possitive_document_embedding(logger,args.output_dir,checkpoint_step,possitive_training_passage_id_embidx,emb_prefix="passage_",)
                    assert positive_p_embs.shape[0] == len(possitive_training_passage_id)
                    D, I = index.search(positive_p_embs, args.topk_training_d2q) 
                    positive_p_embs = None
                    del positive_p_embs

                index.reset()

                logger.info("***** Finish ANN searching for dual d2q task, construct  *****")

                passage_negative_queries = GenerateNegativeQueryID(
                    args,
                    possitive_training_passage_id,
                    query_embedding2id,
                    training_query_positive_id_inversed,
                    I,
                    effective_p_id,
                    pid2pos_pas_embs_idxs = possitive_training_passage_id_to_subset_embidx,
                    Scores_nearest_neighbor=D if "multi_chunk" in args.model_type else None)

            logger.info("***** Done ANN searching for negative queries *****")

        query_embedding = query_embedding[q_start_idx:q_end_idx]
        query_embedding2id = query_embedding2id[q_start_idx:q_end_idx]

        logger.info("***** Construct ANN Triplet *****")
        prefix =  "ann_grouped_training_data_" if args.grouping_ann_data > 0  else "ann_training_data_"
        train_data_output_path = os.path.join(
            args.output_dir, prefix + str(output_num))
        
    
        if args.grouping_ann_data > 0 :
            with open(train_data_output_path, 'w') as f:
                query_range = list(range(query_range_number))
                random.shuffle(query_range)
                
                counting=0
                pos_q_group={}
                pos_d_group={}
                neg_D_group={} # {0:[], 1:[], 2:[]...}
                if args.dual_training:
                    neg_Q_group={}
                for query_idx in query_range:
                    query_id = query_embedding2id[query_idx]
                    if query_id not in effective_q_id or query_id not in training_query_positive_id:
                        continue
                    pos_pid = training_query_positive_id[query_id]
                    
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
            with open(train_data_output_path, 'w') as f:
                query_range = list(range(query_range_number))
                random.shuffle(query_range)
                # old version implementation 
                for query_idx in query_range:
                    query_id = query_embedding2id[query_idx]
                    if query_id not in effective_q_id or query_id not in training_query_positive_id:
                        continue
                    pos_pid = training_query_positive_id[query_id]
                    if not args.dual_training:
                        f.write(
                            "{}\t{}\t{}\n".format(
                                query_id, pos_pid,
                                ','.join(
                                    str(neg_pid) for neg_pid in query_negative_passage[query_id])))
                    else:
                        if pos_pid not in effective_p_id or pos_pid not in training_query_positive_id_inversed:
                            continue
                        f.write(
                            "{}\t{}\t{}\t{}\n".format(
                                query_id, pos_pid,
                                ','.join(
                                    str(neg_pid) for neg_pid in query_negative_passage[query_id]),
                                ','.join(
                                    str(neg_qid) for neg_qid in passage_negative_queries[pos_pid])
                            )
                        )
                    
        ndcg_output_path = os.path.join(
            args.output_dir, "ann_ndcg_" + str(output_num))
        
        if args.data_type==0:
            eval_dict_todump['trec2019_ndcg']=trec2019_ndcg
        with open(ndcg_output_path, 'w') as f:
            json.dump(eval_dict_todump, f)

        return None #dev_ndcg, num_queries_dev


def GenerateNegativePassaageID(
        args,
        query_embedding2id,
        passage_embedding2id,
        training_query_positive_id,
        I_nearest_neighbor,
        effective_q_id):
    query_negative_passage = {}
    SelectTopK = args.ann_measure_topk_mrr
    mrr = 0  # only meaningful if it is SelectTopK = True
    num_queries = 0

    for query_idx in range(I_nearest_neighbor.shape[0]):

        query_id = query_embedding2id[query_idx]

        if query_id not in effective_q_id:
            continue

        num_queries += 1

        pos_pid = training_query_positive_id[query_id]
        top_ann_pid = I_nearest_neighbor[query_idx, :].copy()

        if SelectTopK:
            selected_ann_idx = top_ann_pid[:args.negative_sample + 1]
        else:
            negative_sample_I_idx = list(range(I_nearest_neighbor.shape[1]))
            random.shuffle(negative_sample_I_idx)
            selected_ann_idx = top_ann_pid[negative_sample_I_idx]

        query_negative_passage[query_id] = []

        neg_cnt = 0
        rank = 0

        for idx in selected_ann_idx:
            neg_pid = passage_embedding2id[idx]
            rank += 1
            if neg_pid == pos_pid:
                if rank <= 10:
                    mrr += 1 / rank
                continue

            if neg_pid in query_negative_passage[query_id]:
                # ignore duplicate negative passages for multi_chunk maxP model
                continue

            if neg_cnt >= args.negative_sample:
                break

            query_negative_passage[query_id].append(neg_pid)
            neg_cnt += 1

    if SelectTopK:
        print("Rank:" + str(args.rank) +
              " --- ANN MRR:" + str(mrr / num_queries))

    return query_negative_passage

def GenerateNegativeQueryID(
        args,
        passage_embedding2id,
        query_embedding2id,
        training_query_positive_id_inversed,
        I_nearest_neighbor,
        effective_p_id,
        pid2pos_pas_embs_idxs=None,
        Scores_nearest_neighbor=None):
    passage_negative_query = {}
    SelectTopK = args.ann_measure_topk_mrr
    mrr = 0  # only meaningful if it is SelectTopK = True
    num_passages = 0
    
    # for passage_idx in range(I_nearest_neighbor.shape[0]):
    #     passage_id = passage_embedding2id[passage_idx]
    retrieval_top_k = I_nearest_neighbor.shape[1]

    for passage_id in pid2pos_pas_embs_idxs:

        if passage_id not in effective_p_id:
            continue
        num_passages += 1
        pos_qid = training_query_positive_id_inversed[passage_id]
        
        if Scores_nearest_neighbor is not None:
            # handle multi chunk MaxP model
            top_ann_qidx=[]
            top_ann_qidx_score=[]
            # TODO check
            # merge retrieval result for each positive passage
            for j in range(pid2pos_pas_embs_idxs[passage_id].shape[0]):
                p_idx = pid2pos_pas_embs_idxs[passage_id][j]
                top_ann_qidx.append(I_nearest_neighbor[p_idx, :].copy()) # [topk,]
                top_ann_qidx_score.append(Scores_nearest_neighbor[p_idx, :].copy())
            top_ann_qidx=np.concatenate(top_ann_qidx,axis=0) # [n_MaxP_chunk * topk,]
            top_ann_qidx_score=np.concatenate(top_ann_qidx_score,axis=0) # [n_MaxP_chunk * topk,]
            assert top_ann_qidx.shape == top_ann_qidx_score.shape
            assert top_ann_qidx.shape[0]== pid2pos_pas_embs_idxs[passage_id].shape[0]*retrieval_top_k
            top_ann_qidx = top_ann_qidx[np.argsort(top_ann_qidx_score)[::-1]] # [n_MaxP_chunk * topk,], qidxs in score descending order, but with duplicates
            _, unique_idxs = np.unique(top_ann_qidx,return_index=True)
            unique_idxs = sorted(unique_idxs)
            top_ann_qidx = top_ann_qidx[unique_idxs][:retrieval_top_k]# remove duplicates and keep the topk
            
        else:
            assert pid2pos_pas_embs_idxs[passage_id].shape[0] == 1
            passage_idx = pid2pos_pas_embs_idxs[passage_id][0]
            top_ann_qidx = I_nearest_neighbor[passage_idx, :].copy()

        if SelectTopK:
            selected_ann_idx = top_ann_qidx[:args.negative_sample + 1]
        else:
            negative_sample_I_idx = list(range(I_nearest_neighbor.shape[1]))
            random.shuffle(negative_sample_I_idx)
            selected_ann_idx = top_ann_qidx[negative_sample_I_idx]

        passage_negative_query[passage_id] = []

        neg_cnt = 0
        rank = 0

        for idx in selected_ann_idx:
            neg_qid = query_embedding2id[idx]
            rank += 1
            if neg_qid in pos_qid:
                if rank <= 10:
                    mrr += 1 / rank
                continue

            if neg_qid in passage_negative_query[passage_id]:
                # ignore duplicate negative passages for multi_chunk maxP model
                continue

            if neg_cnt >= args.negative_sample:
                break

            passage_negative_query[passage_id].append(neg_qid)
            neg_cnt += 1

    if SelectTopK:
        print("Rank:" + str(args.rank) +
              " --- ANN MRR:" + str(mrr / num_passages))

    return passage_negative_query


def EvalDevQuery(
        args,
        query_embedding2id,
        passage_embedding2id,
        dev_query_positive_id,
        I_nearest_neighbor):
    # [qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)
    prediction = {}

    for query_idx in range(I_nearest_neighbor.shape[0]):
        query_id = query_embedding2id[query_idx]
        prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx, :].copy()
        selected_ann_idx = top_ann_pid[:50]
        rank = 0
        seen_pid = set()
        for idx in selected_ann_idx:
            pred_pid = passage_embedding2id[idx]

            if pred_pid not in seen_pid:
                # this check handles multiple vector per document
                rank += 1
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)

    # use out of the box evaluation script
    evaluator = pytrec_eval.RelevanceEvaluator(
        convert_to_string_id(dev_query_positive_id), {'map_cut', 'ndcg_cut'})

    eval_query_cnt = 0
    result = evaluator.evaluate(convert_to_string_id(prediction))
    ndcg = 0

    for k in result.keys():
        eval_query_cnt += 1
        ndcg += result[k]["ndcg_cut_10"]

    final_ndcg = ndcg / eval_query_cnt
    print("Rank:" + str(args.rank) + " --- ANN NDCG@10:" + str(final_ndcg))

    return final_ndcg, eval_query_cnt


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
        "--data_type", 
        type=int, 
        default=0, 
        help="the length of new model",
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
        help="Model type selected in the list: " +
        ", ".join(
            MSMarcoConfigDict.keys()),
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
        default=10000,
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
        default=5,  # for 500k queryes, divided into 100k chunks for each epoch
        type=int,
        help="devide training queries into chunks",
    )

    parser.add_argument(
        "--topk_training",
        default=500,
        type=int,
        help="top k from which negative samples are collected",
    )

    parser.add_argument(
        "--topk_training_d2q",
        default=200,
        type=int,
        help="top k from which negative samples are collected",
    )

    parser.add_argument(
        "--negative_sample",
        default=5,
        type=int,
        help="at each resample, how many negative samples per query do I use",
    )

    parser.add_argument(
        "--ann_measure_topk_mrr",
        default=False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--only_keep_latest_embedding_file",
        default=False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA when available",
    )
    parser.add_argument(
        "--not_load_model_for_inference",
        default=False,
        action="store_true",
        help="if we only need to load the embedding, we don't need the models",
    )
    
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="For distributed training: local_rank",
    )

    parser.add_argument(
        "--faiss_omp_num_threads", 
        type=int, 
        default=16, 
        help="for faiss.omp_set_num_threads()",
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
    
    parser.add_argument(
        "--inference",
        default=False,
        action="store_true",
        help="only do inference if specify",
    )

    parser.add_argument(
        "--save_training_query_trec",
        default=False,
        action="store_true",
        help="..",
    )
    # ----------------------------------------------------------------
    parser.add_argument(
        "--dual_training",
        action="store_true",
        help="enable dual training, change the data loading, forward function and loss function",
    )

    # ------------------- L2 normalization ------------------------
    parser.add_argument(
        "--representation_l2_normalization",
        action="store_true",
        help="enable l2_normalization on the representative embeddings for ANN retrieval",
    )

    parser.add_argument(
        "--grouping_ann_data",
        type=int, 
        default=-1,         
        help="group multiple <q,d> pair data into one line, I prefer set to 32",
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
        "--dev_split_num",
        type=int, 
        default=-1,         
        help="how much fold to split validation set",
    )

    parser.add_argument(
        "--testing_split_idx",
        type=int, 
        default=0,         
        help="how much fold to split validation set",
    )

    parser.add_argument(
        "--query_likelihood_strategy",
        type=str, 
        default="positive_doc",
        choices=["BM25_retrieval","positive_doc"],       
        help="use what doc to do retrieval",
    )    
    parser.add_argument(
        "--bm25_top_d2q_path",
        type=str, 
        default="../data/document_BM25/positive_documents.train.truncated_1000.top1000_queries.ANCE_id.tsv",
        help="retrieval results",
    ) 
    parser.add_argument(
        "--d2q_task_evaluation",
        action="store_true",
        help="evaluate and print out the d->q retrieval results",
    ) 
    parser.add_argument(
        "--d2q_task_marco_dev_qrels",
        type=str,
        default=None,
        help="reversed d2q_qrels.tsv, if split validation, should proved a file like: args.d2q_task_marco_dev_qrels+2_fold.split_dict ",
    ) 
    
    args = parser.parse_args()

    return args


def set_env(args):
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


def ann_data_gen(args):
    last_checkpoint = args.last_checkpoint_dir
    ann_no, ann_path, ndcg_json = get_latest_ann_data(args.output_dir)
    output_num = ann_no + 1

    logger.info("starting output number %d", output_num)

    if is_first_worker():
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)

    training_positive_id, dev_positive_id = load_positive_ids(args)

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
            generate_new_ann(
                args,
                output_num,
                next_checkpoint,
                training_positive_id,
                dev_positive_id,
                latest_step_num)
            if args.inference:
                break
            logger.info("finished generating ann data number %d", output_num)
            output_num += 1
            last_checkpoint = next_checkpoint
        if args.local_rank != -1:
            dist.barrier()
        time.sleep(60)


def main():
    args = get_arguments()
    set_env(args)
    ann_data_gen(args)


if __name__ == "__main__":
    main()
