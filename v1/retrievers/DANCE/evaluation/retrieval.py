from enum import Flag
import glob
import sys
sys.path += ['../utils']
import csv
from tqdm import tqdm 
import collections
import gzip
import pickle
import numpy as np
import faiss
import os
import pytrec_eval
import json
from msmarco_eval import quality_checks_qids, compute_metrics, load_reference
import time
import logging
from metric import Metric
import argparse


def get_gpu_index(cpu_index):
    gpu_resources = []
    ngpu = faiss.get_num_gpus()
    tempmem = -1
    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        if tempmem >= 0:
            res.setTempMemory(tempmem)
        gpu_resources.append(res)

    def make_vres_vdev(i0=0, i1=-1):
        " return vectors of device ids and resources useful for gpu_multiple"
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        if i1 == -1:
            i1 = ngpu
        for i in range(i0, i1):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        return vres, vdev

    co = faiss.GpuMultipleClonerOptions()
    co.shard = True 
    gpu_vector_resources, gpu_devices_vector = make_vres_vdev(0, ngpu)
    gpu_index = faiss.index_cpu_to_gpu_multiple(gpu_vector_resources, gpu_devices_vector, cpu_index, co)
    return gpu_index

def document_split_faiss_index(emb_dir,checkpoint_postfix,dev_query_emb,gpu_index=False,top_k=1000,emb_prefix="passage_"):
    logging.info(f"***** processing faiss indexing in split-mode *****")

    emb_file_pattern = os.path.join(emb_dir,f'{emb_prefix}{checkpoint_postfix}__emb_p__data_obj_*.pb')
    emb_file_lists = glob.glob(emb_file_pattern)
    emb_file_lists = sorted(emb_file_lists, key=lambda name: int(name.split('_')[-1].replace('.pb',''))) # sort with split num
    logging.info(f"pattern {emb_file_pattern}\n file lists: {emb_file_lists}")

    # [[scores-nparray,scores-nparray..],[ANCE_ids-nparray,ANCE_ids-nparray,...]]
    merged_candidate_pair_dev = {"D":None,"I":None}

    top_k_dev = top_k
    
    index_offset=0

    for emb_file in emb_file_lists:
        with open(emb_file,'rb') as handle:
            sub_passage_embedding = pickle.load(handle)
        # embid_file = emb_file.replace('emb_p','embid_p')
        # with open(embid_file,'rb') as handle:
        #     sub_passage_embedding2id = pickle.load(handle)
        logging.info(f"loaded {emb_file} embeddings")
        logging.info(f"sub_passage_embedding size {sub_passage_embedding.shape}")
        dim = sub_passage_embedding.shape[1]
        faiss.omp_set_num_threads(16)
        cpu_index = faiss.IndexFlatIP(dim)
        logging.info("***** Faiss: total {} gpus *****".format(faiss.get_num_gpus()))
        index = get_gpu_index(cpu_index) if gpu_index else cpu_index
        index.add(sub_passage_embedding)        

        D, dev_I = index.search(dev_query_emb, top_k_dev) # [n_dev,top]; [n_dev,top]
        dev_I = dev_I + index_offset
        if merged_candidate_pair_dev["D"] is None:
            merged_candidate_pair_dev["D"] = D
            merged_candidate_pair_dev["I"] = dev_I
        else:
            merged_candidate_pair_dev["D"] = np.concatenate([merged_candidate_pair_dev["D"],D],axis=1) # [n_dev,topk_dev *2]
            merged_candidate_pair_dev["I"] = np.concatenate([merged_candidate_pair_dev["I"],dev_I],axis=1) # [n_dev,topk_dev *2]
            sorted_ind = np.flip(np.argsort(merged_candidate_pair_dev["D"],axis=1),axis=1) # descent sort along topk_dev *2 scores for each row in n_dev
            merged_candidate_pair_dev["D"]=np.take_along_axis(merged_candidate_pair_dev["D"], sorted_ind, axis=1)[:,:top_k_dev] # [n_dev,topk_dev *2]
            merged_candidate_pair_dev["I"]=np.take_along_axis(merged_candidate_pair_dev["I"], sorted_ind, axis=1)[:,:top_k_dev] # [n_dev,topk_dev *2]
    
        index_offset = index_offset + sub_passage_embedding.shape[0]
        index.reset()
        sub_passage_embedding=None
    
    return merged_candidate_pair_dev["D"],merged_candidate_pair_dev["I"]

def save_trec_file(query_embedding2id, passage_embedding2id,
                I_nearest_neighbor,D_nearest_neighbor,
                trec_save_path,topN=1000):

    print(f"saving trec file to {trec_save_path}")
    output_f = open(trec_save_path,"w")
    # qids_to_ranked_candidate_passages = {} 

    # select_top_K = topN if "MaxP" not in task_name else  topN*4
    select_top_K = topN

    for query_idx in range(len(I_nearest_neighbor)): 
        seen_pid = set() # handle MaxP
        query_id = query_embedding2id[query_idx]
        # prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx].copy()
        selected_ann_idx = top_ann_pid[:select_top_K]

        rank = 0
        for matrix_rank, idx in enumerate(selected_ann_idx):
            pred_pid = passage_embedding2id[idx]
            
            if not pred_pid in seen_pid:
                # this check handles multiple vector per document
                output_f.write(f"{str(query_id)} Q0 {str(pred_pid)} {str(rank+1)} {str(D_nearest_neighbor[query_idx][matrix_rank])} ance\n")
                rank += 1
                seen_pid.add(pred_pid)

                if len(seen_pid) >= topN:
                    break

    output_f.close()

# memory saving embedding loading
def get_p_emb_idx_by_pid_list(pid_list,embbeding_dir,checkpoint_step,emb_prefix="passage_"):
    emb_file_pattern = os.path.join(embbeding_dir,f'{emb_prefix}{checkpoint_step}__emb_p__data_obj_*.pb')
    emb_file_lists = glob.glob(emb_file_pattern)
    emb_file_lists = sorted(emb_file_lists, key=lambda name: int(name.split('_')[-1].replace('.pb',''))) # sort with split num
    

    embedding2id = []
    for emb_file in emb_file_lists:
        embid_file= emb_file.replace('emb_p','embid_p')
        with open(embid_file,'rb') as handle:
            embedding2id.append(pickle.load(handle))
        
    embedding2id = np.concatenate(embedding2id, axis=0)

    pid_idxs=[]
    p_emb2id=[]
    for pid in pid_list:
        p_idx=np.asarray(np.where(embedding2id==pid)).flatten()
        pid_idxs.append(p_idx)
        p_emb2id.extend([pid for _ in range(p_idx.shape[0])])
    pid_idxs = np.concatenate(pid_idxs,axis=0)
    p_emb2id = np.asarray(p_emb2id).flatten()

    assert p_emb2id.shape[0] == pid_idxs.shape[0]

    return pid_idxs.reshape([-1,]), p_emb2id

def split_load_doc_emb_by_pid_embidx_list(pid_embidx_list,embbeding_dir,checkpoint_step,emb_prefix="passage_"):
    
    pid_embidx_list=np.asarray(pid_embidx_list).reshape([-1,])
    
    emb_file_pattern = os.path.join(embbeding_dir,f'{emb_prefix}{checkpoint_step}__emb_p__data_obj_*.pb')
    emb_file_lists = glob.glob(emb_file_pattern)
    emb_file_lists = sorted(emb_file_lists, key=lambda name: int(name.split('_')[-1].replace('.pb',''))) # sort with split num
    
    idx_lower_bound=0
    p_embedding=np.ones([pid_embidx_list.shape[0],768])
    for emb_file in emb_file_lists:
        with open(emb_file,'rb') as handle:
            embedding=pickle.load(handle)
            dt=embedding.dtype

        idx_upper_bound = idx_lower_bound + embedding.shape[0]
        sub_relative_idx = np.intersect1d(np.where(pid_embidx_list>=idx_lower_bound),
                                np.where(pid_embidx_list<idx_upper_bound)) 

        p_embedding[sub_relative_idx]=embedding[pid_embidx_list[sub_relative_idx] - idx_lower_bound]
        idx_lower_bound += embedding.shape[0]

    assert p_embedding.shape[0] == pid_embidx_list.shape[0]
    return p_embedding.astype(dt)

def load_passage_embeddings(load_emb=True,emb_prefix='passage_'):
    print(f'loading passage embeddings {load_emb} and ids')
    emb_file_pattern = os.path.join(checkpoint_path,f'{emb_prefix}{checkpoint}__emb_p__data_obj_*.pb')
    emb_file_lists = glob.glob(emb_file_pattern)
    emb_file_lists = sorted(emb_file_lists, key=lambda name: int(name.split('_')[-1].replace('.pb',''))) # sort with split num
    print(f"pattern {emb_file_pattern}\n file lists: {emb_file_lists}")

    passage_embedding = []
    passage_embedding2id = []
    for emb_file in emb_file_lists:
        try:
            if load_emb:
                with open(emb_file,'rb') as handle:
                    passage_embedding.append(pickle.load(handle))
            embid_file = emb_file.replace('emb_p','embid_p')
            with open(embid_file,'rb') as handle:
                passage_embedding2id.append(pickle.load(handle))
        except:
            break

    if load_emb:
        passage_embedding = np.concatenate(passage_embedding, axis=0)
    passage_embedding2id = np.concatenate(passage_embedding2id, axis=0)
    if load_emb:
        print(f'passge embedding shape {passage_embedding.shape}, passage offset-id shape {passage_embedding2id.shape}')
    print(f'passge embedding [:3] {passage_embedding[:3]}, passage offset-id [:3] {passage_embedding2id[:3]}')    

    return passage_embedding, passage_embedding2id


def convert_trec_to_MARCO_id(processed_data_dir,tmp_path,output_path,queryset_prefix,data_type=0,d2q_trec=False):

    # trec_save_path_list = glob.glob(f"{task_name}data-type-{data_type}_test-set-{test_set}_ckpt-{specified_checkpoint}.trec")

    qidmap_path = os.path.join(processed_data_dir,"{}-query_qid2offset.pickle".format(queryset_prefix))
    with open(qidmap_path, 'rb') as handle:
        qid2offset = pickle.load(handle)
    pidmap_path = os.path.join(processed_data_dir,"pid2offset.pickle")
    with open(pidmap_path, 'rb') as handle:
        pid2offset = pickle.load(handle)

    offset2qid = {}
    for k in qid2offset:
        offset2qid[qid2offset[k]]=k
    offset2pid = {}
    for k in pid2offset:
        offset2pid[pid2offset[k]]=k

    # for path in trec_save_path_list:
    print("processing file:",tmp_path)
    with open(tmp_path) as f:
        lines=f.readlines()
    with open(output_path,"w") as f:
        for line in tqdm(lines):
            qid , Q0, pid, rank, score, tag = line.strip().split(' ')
            # print(offset2qid[int(qid)] , Q0, pid, rank, score.replace('-',''), tag)
            if not d2q_trec:
                qid_inrun=str(offset2qid[int(qid)])
                pid_inrun="D"+str(offset2pid[int(pid)]) if data_type==0 or data_type==-1 else str(offset2pid[int(pid)])
            else:
                qid_inrun="D"+str(offset2pid[int(qid)]) if data_type==0 or data_type==-1 else str(offset2pid[int(qid)])
                pid_inrun=str(offset2qid[int(pid)])

            f.write(f"{qid_inrun} Q0 {pid_inrun} {rank} {score} {tag}\n")

def retrieval_specify(processed_data_dir,queryset_prefix,trec_save_path,emb_dir,checkpoint_postfix,data_type,topN,gpu_index):

    # trec_save_path = f"{task_name}_data-type-{data_type}_test-set-{test_set}_ckpt-{checkpoint}.trec"
    dev_query_embedding = []
    dev_query_embedding2id = []
    
    for i in range(100):
        try:
            # load the embedding of queries
            with open(
                os.path.join(emb_dir,"{}_query_{}__emb_p__data_obj_{}.pb".format(queryset_prefix,str(checkpoint_postfix),str(i))),
                 'rb') as handle:
                dev_query_embedding.append(pickle.load(handle))
            
            with open(
                os.path.join(emb_dir,"{}_query_{}__embid_p__data_obj_{}.pb".format(queryset_prefix,str(checkpoint_postfix),
                str(i))), 'rb') as handle:
                dev_query_embedding2id.append(pickle.load(handle))
        except:
            break
    if (not dev_query_embedding) or (not dev_query_embedding2id):
        print("No query data found for checkpoint: ",emb_dir)
    dev_query_embedding = np.concatenate(dev_query_embedding, axis=0)
    dev_query_embedding2id = np.concatenate(dev_query_embedding2id, axis=0)

    print(f'query embedding shape {dev_query_embedding.shape}, query offset-id shape {dev_query_embedding2id.shape}')
    print(f'query embedding [:3] {dev_query_embedding[:3]}, query offset-id [:3] {dev_query_embedding2id[:3]}')
    
    # # full ranking metrics

    dev_D, dev_I = document_split_faiss_index(dev_query_embedding,top_k=topN,gpu_index=gpu_index)
    
    _,passage_embedding2id=load_passage_embeddings(load_emb=False,emb_prefix='passage_')
    save_trec_file(dev_query_embedding2id, passage_embedding2id,
                    dev_I,dev_D,"tmp_intermediate.trec",topN=topN)

    convert_trec_to_MARCO_id(
        processed_data_dir=processed_data_dir,
        tmp_path="tmp_intermediate.trec", 
        output_path=trec_save_path,
        test_set=queryset_prefix,
        data_type=data_type)
    os.remove("tmp_intermediate.trec")
    return None

def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--emb_dir",
        default="../task_folder/ann/",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--checkpoint_postfix",
        default= "1000",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--emb_dir",
        default="../task_folder/ann/",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--processed_data_dir",
        default="../data/preprocessed_data",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--trec_path",
        default="./result.trec",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--queryset_prefix",
        default="dev",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data_type", 
        type=int, 
        default=0, 
        help="document or passage",
    )
    parser.add_argument(
        "--topN", 
        type=int, 
        default=100, 
        help="document or passage",
    )
    parser.add_argument(
        "--gpu_index", 
        type=bool, 
        default=False, 
        help="document or passage",
    )
    args = parser.parse_args()

    return args

def main():

    # task_name = "custom name"
    # emb_dir = "../task_folder/ann/"# location for dumpped query and passage/document embeddings which is output_dir 
    # checkpoint_postfix = "1000"
    # data_type = 0 # 0 for document, 1 for passage    
    # processed_data_dir = "../data/preprocessed_data"# preprocessed data folder
    # queryset_prefix = "dev"
    # gpu_index=False
    # topN = 100
    # # raw_data_dir = "../data/raw_data/" # folder for the qrels

    args = get_arguments()
    emb_dir = args.emb_dir # location for dumpped query and passage/document embeddings which is output_dir 
    checkpoint_postfix = args.checkpoint_postfix
    data_type = args.data_type # 0 for document, 1 for passage    
    processed_data_dir = args.processed_data_dir # preprocessed data folder
    trec_path = args.trec_path
    queryset_prefix = args.queryset_prefix 
    gpu_index = args.gpu_index
    topN = args.topN

    print("start time:",time.asctime( time.localtime(time.time()) ))
    # trec_save_path = f"{task_name}_data-type-{data_type}_query-set-{queryset_prefix}_ckpt-{checkpoint_postfix}.trec"
    retrieval_specify(
        processed_data_dir=processed_data_dir,
        queryset_prefix=queryset_prefix,trec_save_path=trec_path,
        emb_dir=emb_dir,checkpoint_postfix=checkpoint_postfix,data_type=data_type,topN=topN,gpu_index=gpu_index)


if __name__ == '__main__':
    main()

