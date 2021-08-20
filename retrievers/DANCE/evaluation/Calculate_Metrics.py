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

task_name='DANCE' # 
checkpoint_path = # location for dumpped query and passage/document embeddings which is output_dir 
checkpoint =  0 # embedding from which checkpoint(ie: 400000)
data_type = 0 # 0 for document, 1 for passage
test_sets = [0]  # 0 for dev_set, 1 for eval_set, 'docleaderboard' for leader board
raw_data_dir = "../data/raw_data/" # folder for the qrels
processed_data_dir = # preprocessed data folder

print("start time:",time.asctime( time.localtime(time.time()) ))
# # Load Qrel
split_ann_search=True
gpu_index=True
is_metric_retrieval=True
is_d2q_retrieval=True
is_q2d_retrieval=True
format_to_MACRO_id=True

dev_split_num = -1

from metric import Metric


if data_type == 0:
    topN = 200
else:
    topN = 200


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

def document_split_faiss_index(dev_query_emb,top_k=1000,emb_prefix="passage_"):
    logging.info(f"***** processing faiss indexing in split-mode *****")

    emb_file_pattern = os.path.join(checkpoint_path,f'{emb_prefix}{checkpoint}__emb_p__data_obj_*.pb')
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

def convert_to_string_id(result_dict):
    string_id_dict = {}

    # format [string, dict[string, val]]
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v

        string_id_dict[str(k)] = _temp_v

    return string_id_dict
def save_trec_file(query_embedding2id, passage_embedding2id,
                I_nearest_neighbor,D_nearest_neighbor,
                trec_save_path,topN=1000):

    print(f"saving trec file to {trec_save_path}")
    output_f = open(trec_save_path,"w")
    # qids_to_ranked_candidate_passages = {} 

    select_top_K = topN if "MaxP" not in task_name else  topN*4

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

def evaluation(data_type,trec_path,dev_query_embedding2id,passage_embedding2id,dev_I,dev_D,test_set="trec2019",dev_split_num=-1,split_idx=-1):
    if data_type ==0 :
        if  test_set== "marcodev":
            qrels="../data/raw_data/msmarco-docdev-qrels.tsv"
        elif test_set== "trec2019":
            qrels="../data/raw_data/2019qrels-docs.txt"
    elif data_type ==1:
        if test_set == "marcodev":
            qrels="../data/raw_data/qrels.dev.small.tsv"
    else:
        logging.error("wrong data type")
        exit()

    trec_path=trec_path.replace(".trec",".formatted.trec")
    met = Metric()
    if split_idx >= 0:
        split_file_path=qrels+f"{dev_split_num}_fold.split_dict"
        with open(split_file_path,'rb') as f:
            split=pickle.load(f)
    else:
        split=None
    
    ndcg10 = met.get_metric(qrels, trec_path, 'ndcg_cut_10',split,split_idx)
    mrr10 = met.get_mrr(qrels, trec_path, 'mrr_cut_10',split,split_idx)
    mrr100 = met.get_mrr(qrels, trec_path, 'mrr_cut_100',split,split_idx)

    print(f" evaluation for {test_set}, trec_file {trec_path}, split_idx {split_idx} \
        ndcg_cut_10 : {ndcg10}, \
        mrr_cut_10 : {mrr10}, \
        mrr_cut_100 : {mrr100}"
    )

    return ndcg10

def EvalDevQuery(query_embedding2id, passage_embedding2id, dev_query_positive_id, I_nearest_neighbor,topN ):
    prediction = {} #[qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)

    total = 0
    labeled = 0
    Atotal = 0
    Alabeled = 0
    qids_to_ranked_candidate_passages = {} 
    for query_idx in range(len(I_nearest_neighbor)): 
        seen_pid = set()
        query_id = query_embedding2id[query_idx]
        prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        rank = 0
        
        if query_id in qids_to_ranked_candidate_passages:
            pass    
        else:
            # By default, all PIDs in the list of 1000 are 0. Only override those that are given
            tmp = [0] * 1000
            qids_to_ranked_candidate_passages[query_id] = tmp
                
        for idx in selected_ann_idx:
            pred_pid = passage_embedding2id[idx]
            
            if not pred_pid in seen_pid:
                # this check handles multiple vector per document
                qids_to_ranked_candidate_passages[query_id][rank]=pred_pid
                Atotal += 1
                if pred_pid not in dev_query_positive_id[query_id]:
                    Alabeled += 1
                if rank < 10:
                    total += 1
                    if pred_pid not in dev_query_positive_id[query_id]:
                        labeled += 1
                rank += 1
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)

    # use out of the box evaluation script
    evaluator = pytrec_eval.RelevanceEvaluator(
        convert_to_string_id(dev_query_positive_id), {'map_cut', 'ndcg_cut', 'recip_rank','recall'})

    eval_query_cnt = 0
    result = evaluator.evaluate(convert_to_string_id(prediction))
    
    qids_to_relevant_passageids = {}
    for qid in dev_query_positive_id:
        qid = int(qid)
        if qid in qids_to_relevant_passageids:
            pass
        else:
            qids_to_relevant_passageids[qid] = []
            for pid in dev_query_positive_id[qid]:
                if pid>0:
                    qids_to_relevant_passageids[qid].append(pid)
            
    ms_mrr = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)

    ndcg = 0
    Map = 0
    mrr = 0
    recall = 0
    recall_1000 = 0

    for k in result.keys():
        eval_query_cnt += 1
        ndcg += result[k]["ndcg_cut_10"]
        Map += result[k]["map_cut_10"]
        mrr += result[k]["recip_rank"]
        recall += result[k]["recall_"+str(topN)]

    final_ndcg = ndcg / eval_query_cnt
    final_Map = Map / eval_query_cnt
    final_mrr = mrr / eval_query_cnt
    final_recall = recall / eval_query_cnt
    hole_rate = labeled/total
    Ahole_rate = Alabeled/Atotal

    return final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, result, prediction

def convert_trec_to_MARCO_id(path,data_type,test_set,task_name,d2q_trec=False):

    specified_checkpoint=checkpoint

    # if test_set=="training":
    #     ann_data_dir="../data/off_ckpt/ann_data_official_Doc_FirstP_512/"
    #     trec_save_path_list = glob.glob(os.path.join(ann_data_dir,"ann_training_query_retrieval_*.trec"))
    # elif test_set=="docleaderboard":
    #     trec_save_path_list = glob.glob(f"*data-type-{data_type}_test-set-{test_set}_ckpt-*.trec")
    # else:
    if task_name is not None:
        task_name = task_name + '_'
    if specified_checkpoint is not None:
        specified_checkpoint = str(specified_checkpoint)
    else:
        specified_checkpoint = '*'

    # trec_save_path_list = glob.glob(f"{task_name}data-type-{data_type}_test-set-{test_set}_ckpt-{specified_checkpoint}.trec")

    if test_set=="training":
        with open(os.path.join(processed_data_dir,'train-query_qid2offset.pickle'),'rb') as f:
            qid2offset = pickle.load(f)
    elif test_set=="docleaderboard":
        with open(os.path.join(processed_data_dir,'docleaderboard_qid2offset.pickle'),'rb') as f:
            qid2offset = pickle.load(f)
    else:
        if test_set==1:
            with open(os.path.join(processed_data_dir,'dev-query_qid2offset.pickle'),'rb') as f:
                qid2offset = pickle.load(f)
        else:
            with open(os.path.join(processed_data_dir,'real-dev-query_qid2offset.pickle'),'rb') as f:
                qid2offset = pickle.load(f)
    offset2qid = {}
    for k in qid2offset:
        offset2qid[qid2offset[k]]=k

    with open(os.path.join(processed_data_dir,'pid2offset.pickle'),'rb') as f:
        pid2offset = pickle.load(f)
    offset2pid = {}
    for k in pid2offset:
        offset2pid[pid2offset[k]]=k

    # for path in trec_save_path_list:
    print("processing file:",path)
    with open(path) as f:
        lines=f.readlines()
    with open(path.replace(".trec",".formatted.trec"),"w") as f:
        for line in tqdm(lines):
            qid , Q0, pid, rank, score, tag = line.strip().split(' ')
            # print(offset2qid[int(qid)] , Q0, pid, rank, score.replace('-',''), tag)
            if not d2q_trec:
                qid_inrun=str(offset2qid[int(qid)])
                pid_inrun="D"+str(offset2pid[int(pid)]) if data_type==0 else str(offset2pid[int(pid)])
            else:
                qid_inrun="D"+str(offset2pid[int(qid)]) if data_type==0 else str(offset2pid[int(qid)])
                pid_inrun=str(offset2qid[int(pid)])

            f.write(f"{qid_inrun} Q0 {pid_inrun} {rank} {score} {tag}\n")

def retrieval(passage_embedding,passage_embedding2id,test_set,index,task_name):

    trec_save_path = f"{task_name}_data-type-{data_type}_test-set-{test_set}_ckpt-{checkpoint}.trec"

    if test_set == 1 or  test_set ==0:

        if test_set == 1:
            qidmap_path = processed_data_dir+"/dev-query_qid2offset.pickle"
        else:
            qidmap_path = processed_data_dir+"/real-dev-query_qid2offset.pickle"

        if data_type == 0:
            if test_set == 1:
                query_path = raw_data_dir+"/msmarco-test2019-queries.tsv"
                passage_path = raw_data_dir+"/msmarco-doctest2019-top100"
            else:
                query_path = raw_data_dir+"/msmarco-docdev-queries.tsv"
                passage_path = raw_data_dir+"/msmarco-docdev-top100"
        else:
            if test_set == 1:
                query_path = raw_data_dir+"/msmarco-test2019-queries.tsv"
                passage_path = raw_data_dir+"/msmarco-passagetest2019-top1000.tsv"
            else:
                query_path = raw_data_dir+"/queries.dev.small.tsv"
                passage_path = raw_data_dir+"/top1000.dev.tsv"


        with open(qidmap_path, 'rb') as handle:
            qidmap = pickle.load(handle)


        qset = set()
        with gzip.open(query_path, 'rt', encoding='utf-8') if query_path[-2:] == "gz" else open(query_path, 'rt', encoding='utf-8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [qid, query] in tsvreader:
                qset.add(qid)

        pidmap_path = processed_data_dir+"/pid2offset.pickle"
        with open(pidmap_path, 'rb') as handle:
            pidmap = pickle.load(handle)

        bm25 = collections.defaultdict(set)
        with gzip.open(passage_path, 'rt', encoding='utf-8') if passage_path[-2:] == "gz" else open(passage_path, 'rt', encoding='utf-8') as f:
            for line in tqdm(f):
                if data_type == 0:
                    [qid, Q0, pid, rank, score, runstring] = line.split(' ')
                    pid = pid[1:]
                else:
                    [qid, pid, query, passage] = line.split("\t")
                if qid in qset and int(qid) in qidmap:
                    bm25[qidmap[int(qid)]].add(pidmap[int(pid)]) 

        print("number of queries with " +str(topN) + " BM25 passages:", len(bm25))


    dev_query_embedding = []
    dev_query_embedding2id = []

    for i in range(100):
        try:
            if test_set=="docleaderboard":
                # load the embedding of testset
                with open(checkpoint_path + test_set+ "_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                    dev_query_embedding.append(pickle.load(handle))
                with open(checkpoint_path + test_set+ "_query_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                    dev_query_embedding2id.append(pickle.load(handle))
            elif test_set == "training":
                with open(checkpoint_path + "query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                    dev_query_embedding.append(pickle.load(handle))
                with open(checkpoint_path + "query_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                    dev_query_embedding2id.append(pickle.load(handle))
            elif test_set == 0:
                with open(checkpoint_path + "real-dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                    dev_query_embedding.append(pickle.load(handle))
                with open(checkpoint_path + "real-dev_query_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                    dev_query_embedding2id.append(pickle.load(handle))
            elif test_set ==1:
                with open(checkpoint_path + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                    dev_query_embedding.append(pickle.load(handle))
                with open(checkpoint_path + "dev_query_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                    dev_query_embedding2id.append(pickle.load(handle))
        except:
            break
    if (not dev_query_embedding) or (not dev_query_embedding2id):
        print("No query data found for checkpoint: ",checkpoint)
    dev_query_embedding = np.concatenate(dev_query_embedding, axis=0)
    dev_query_embedding2id = np.concatenate(dev_query_embedding2id, axis=0)

    print(f'query embedding shape {dev_query_embedding.shape}, query offset-id shape {dev_query_embedding2id.shape}')
    print(f'query embedding [:3] {dev_query_embedding[:3]}, query offset-id [:3] {dev_query_embedding2id[:3]}')
    
    # # full ranking metrics

    search_top_K = topN if "MaxP" not in task_name else topN*4

    if index is not None:
        dev_D, dev_I = index.search(dev_query_embedding, search_top_K)
    else:
        dev_D, dev_I = document_split_faiss_index(dev_query_embedding,top_k=search_top_K)

    save_trec_file(dev_query_embedding2id, passage_embedding2id,
                    dev_I,dev_D,trec_save_path,topN=topN)

    if format_to_MACRO_id:
        convert_trec_to_MARCO_id(path=trec_save_path,data_type=data_type,test_set=test_set,task_name=task_name)

    if data_type ==0:
        if test_set ==1:
            evaluation(data_type,trec_save_path,dev_query_embedding2id,passage_embedding2id,dev_I,dev_D,test_set="trec2019",dev_split_num=-1,split_idx=-1)
        elif test_set ==0:
            if dev_split_num > 1:
                for i in range(dev_split_num):
                    evaluation(data_type,trec_save_path,dev_query_embedding2id,passage_embedding2id,dev_I,dev_D,test_set="marcodev",dev_split_num=dev_split_num,split_idx=i)
            else:
                evaluation(data_type,trec_save_path,dev_query_embedding2id,passage_embedding2id,dev_I,dev_D,test_set="marcodev",dev_split_num=1,split_idx=-1)
    return None


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

def load_qid_to_embs_by_pid_list(pid_list,embbeding_dir,checkpoint_step):
    emb_file_pattern = os.path.join(embbeding_dir,f'passage_{checkpoint_step}__emb_p__data_obj_*.pb')
    emb_file_lists = glob.glob(emb_file_pattern)
    emb_file_lists = sorted(emb_file_lists, key=lambda name: int(name.split('_')[-1].replace('.pb',''))) # sort with split num
    

    embedding = []
    embedding2id = []
    for emb_file in emb_file_lists:
        with open(emb_file,'rb') as handle:
            embedding.append(pickle.load(handle))
        embid_file = emb_file.replace('emb_p','embid_p')
        with open(embid_file,'rb') as handle:
            embedding2id.append(pickle.load(handle))
        
    embedding = np.concatenate(embedding, axis=0)
    embedding2id = np.concatenate(embedding2id, axis=0)

    pid_embs=[]
    for pid in pid_list:
        p_idx=np.where(embedding2id==pid)
        pid_embs.append(embedding[p_idx].reshape([1,-1]))
    pid_embs = np.concatenate(pid_embs,axis=0)
    
    return pid_embs

def load_positive_q2d(query_positive_id_path):
    dev_query_positive_id={}
    with open(query_positive_id_path, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, docid, rel] in tsvreader:
            topicid = int(topicid)
            docid = int(docid)
            if topicid not in dev_query_positive_id:
                dev_query_positive_id[topicid]=[docid]
            else:
                dev_query_positive_id[topicid].append(docid)

    return dev_query_positive_id


def d2q_multichunk_matrix_convert(Scores_nearest_neighbor,I_nearest_neighbor,pid_list,multichunk_p_emb2id,retrieval_top_k):
    
    pid2pos_pas_embs_idxs={}

    for pid in pid_list:
        p_emb_idxs = np.asarray(np.where(multichunk_p_emb2id == pid)).flatten()
        pid2pos_pas_embs_idxs[pid] = p_emb_idxs

    merged_D=[]
    merged_I=[]
    merged_p_emb2id=[]

    for passage_id in pid2pos_pas_embs_idxs:
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
        top_ann_qidx_score = top_ann_qidx_score[np.argsort(top_ann_qidx_score)[::-1]]
        _, unique_idxs = np.unique(top_ann_qidx,return_index=True) # remove duplicates and keep the topk
        unique_idxs=sorted(unique_idxs)
        top_ann_qidx = top_ann_qidx[unique_idxs]
        top_ann_qidx_score = top_ann_qidx_score[unique_idxs]

        merged_D.append(top_ann_qidx_score[:retrieval_top_k].reshape([1,-1]))
        merged_I.append(top_ann_qidx[:retrieval_top_k].reshape([1,-1]))
        merged_p_emb2id.append(passage_id)
    
    
    merged_D = np.concatenate(merged_D,axis=0)
    merged_I = np.concatenate(merged_I,axis=0)
    merged_p_emb2id = np.asarray(merged_p_emb2id)
        
    return merged_D,merged_I,merged_p_emb2id

def d2q_from_dev_q_retrieval(task_name):
    
    dev_qid_to_pids=load_positive_q2d(query_positive_id_path=os.path.join(processed_data_dir,"real-dev-qrel.tsv")) # {qid:[pid,...]}
    # dev_pid_to_qids={}
    dev_pid_list=[]
    for qid in dev_qid_to_pids:
        for pid in dev_qid_to_pids[qid]:
            if pid not in dev_pid_list:
                dev_pid_list.append(pid)
            # if pid not in dev_pid_to_qids:
            #     dev_pid_to_qids[pid]=[qid]
            # else:
            #     dev_pid_to_qids[pid].append(qid)
    dev_pid_emb_idxs, dev_p_emb2id = get_p_emb_idx_by_pid_list(pid_list=dev_pid_list,embbeding_dir=checkpoint_path,checkpoint_step=checkpoint,emb_prefix="passage_")
    assert (dev_pid_emb_idxs.shape[0] == len(dev_pid_list)) or (dev_pid_emb_idxs.shape[0] == len(dev_pid_list)*4)
    dev_pid_embs=split_load_doc_emb_by_pid_embidx_list(pid_embidx_list=dev_pid_emb_idxs,embbeding_dir=checkpoint_path,checkpoint_step=checkpoint,emb_prefix="passage_")
    assert dev_pid_emb_idxs.shape[0] == dev_pid_embs.shape[0]

    query_embedding,query_embedding2id = load_passage_embeddings(load_emb=True,emb_prefix='real-dev_query_') # retrieval in dev
    print('constructing faiss index')
    dim = query_embedding.shape[1]
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(dim)
    print("***** Faiss: total {} gpus *****".format(faiss.get_num_gpus()))
    index = get_gpu_index(cpu_index) if gpu_index else cpu_index
    index.add(query_embedding)
    
    D, I = index.search(dev_pid_embs,topN)
    if "MaxP" in task_name:
        D, I, dev_p_emb2id = d2q_multichunk_matrix_convert(
            Scores_nearest_neighbor=D,I_nearest_neighbor=I,
            pid_list=dev_pid_list,multichunk_p_emb2id=dev_p_emb2id,retrieval_top_k=topN)
    
    
    trec_save_path = f"{task_name}_data-type-{data_type}_pos-dev-pid_retrieve_top_dev-query_ckpt-{checkpoint}.trec"

    save_trec_file(
        query_embedding2id=dev_p_emb2id,
        passage_embedding2id=query_embedding2id,
        I_nearest_neighbor=I,D_nearest_neighbor=D,
        trec_save_path=trec_save_path,
        topN=topN)
    return trec_save_path

def d2q_from_train_q_retrieval(task_name):
    
    dev_qid_to_pids=load_positive_q2d(query_positive_id_path=os.path.join(processed_data_dir,"real-dev-qrel.tsv")) # {qid:[pid,...]}
    # dev_pid_to_qids={}
    dev_pid_list=[]
    for qid in dev_qid_to_pids:
        for pid in dev_qid_to_pids[qid]:
            if pid not in dev_pid_list:
                dev_pid_list.append(pid)
            # if pid not in dev_pid_to_qids:
            #     dev_pid_to_qids[pid]=[qid]
            # else:
            #     dev_pid_to_qids[pid].append(qid)
    dev_pid_emb_idxs, dev_p_emb2id = get_p_emb_idx_by_pid_list(pid_list=dev_pid_list,embbeding_dir=checkpoint_path,checkpoint_step=checkpoint,emb_prefix="passage_")
    assert (dev_pid_emb_idxs.shape[0] == len(dev_pid_list)) or (dev_pid_emb_idxs.shape[0] == len(dev_pid_list)*4)
    dev_pid_embs=split_load_doc_emb_by_pid_embidx_list(pid_embidx_list=dev_pid_emb_idxs,embbeding_dir=checkpoint_path,checkpoint_step=checkpoint,emb_prefix="passage_")
    assert dev_pid_emb_idxs.shape[0] == dev_pid_embs.shape[0]

    query_embedding,query_embedding2id = load_passage_embeddings(load_emb=True,emb_prefix='query_') # retrieval in dev
    print('constructing faiss index')
    dim = query_embedding.shape[1]
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(dim)
    print("***** Faiss: total {} gpus *****".format(faiss.get_num_gpus()))
    index = get_gpu_index(cpu_index) if gpu_index else cpu_index
    index.add(query_embedding)
    
    D, I = index.search(dev_pid_embs,topN)
    if "MaxP" in task_name:
        D, I, dev_p_emb2id = d2q_multichunk_matrix_convert(
            Scores_nearest_neighbor=D,I_nearest_neighbor=I,
            pid_list=dev_pid_list,multichunk_p_emb2id=dev_p_emb2id,retrieval_top_k=topN)
    
    
    trec_save_path = f"{task_name}_data-type-{data_type}_pos-dev-pid_retrieve_top_train-query_ckpt-{checkpoint}.trec"

    save_trec_file(
        query_embedding2id=dev_p_emb2id,
        passage_embedding2id=query_embedding2id,
        I_nearest_neighbor=I,D_nearest_neighbor=D,
        trec_save_path=trec_save_path,
        topN=topN)   

    return trec_save_path

def q2d_dev_q_retrieval(task_name):
    dev_qid_to_pids=load_positive_q2d(query_positive_id_path=os.path.join(processed_data_dir,"real-dev-qrel.tsv")) # {qid:[pid,...]}
    # dev_pid_to_qids={}
    dev_pid_list=[]
    for qid in dev_qid_to_pids:
        for pid in dev_qid_to_pids[qid]:
            if pid not in dev_pid_list:
                dev_pid_list.append(pid)
            # if pid not in dev_pid_to_qids:
            #     dev_pid_to_qids[pid]=[qid]
            # else:
            #     dev_pid_to_qids[pid].append(qid)
    dev_pid_emb_idxs, dev_p_emb2id=get_p_emb_idx_by_pid_list(pid_list=dev_pid_list,embbeding_dir=checkpoint_path,checkpoint_step=checkpoint,emb_prefix="passage_")
    assert (dev_pid_emb_idxs.shape[0] == len(dev_pid_list)) or (dev_pid_emb_idxs.shape[0] == len(dev_pid_list)*4)
    dev_pid_embs=split_load_doc_emb_by_pid_embidx_list(pid_embidx_list=dev_pid_emb_idxs,embbeding_dir=checkpoint_path,checkpoint_step=checkpoint,emb_prefix="passage_")
    assert dev_pid_emb_idxs.shape[0] == dev_pid_embs.shape[0]

    query_embedding,query_embedding2id = load_passage_embeddings(load_emb=True,emb_prefix='real-dev_query_') # retrieval in dev
    
    print('constructing faiss index')
    dim = dev_pid_embs.shape[1]
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(dim)
    print("***** Faiss: total {} gpus *****".format(faiss.get_num_gpus()))
    index = get_gpu_index(cpu_index) if gpu_index else cpu_index
    index.add(dev_pid_embs)

    search_top_K = topN if "MaxP" not in task_name else topN*4    
    D, I = index.search(query_embedding,search_top_K)

    trec_save_path = f"{task_name}_data-type-{data_type}_pos-dev-qid_retrieve_top_dev-documents_ckpt-{checkpoint}.trec"

    save_trec_file(
        query_embedding2id=query_embedding2id,
        passage_embedding2id=dev_p_emb2id,
        I_nearest_neighbor=I,D_nearest_neighbor=D,
        trec_save_path=trec_save_path,
        topN=topN)
    return trec_save_path


if is_metric_retrieval:
    if not split_ann_search:
        
        passage_embedding,passage_embedding2id=load_passage_embeddings(load_emb=True,emb_prefix='passage_')

        print('constructing faiss index')
        dim = passage_embedding.shape[1]
        faiss.omp_set_num_threads(16)
        cpu_index = faiss.IndexFlatIP(dim)

        print("***** Faiss: total {} gpus *****".format(faiss.get_num_gpus()))
        index = get_gpu_index(cpu_index) if gpu_index else cpu_index
        index.add(passage_embedding)        

        for test_set in test_sets:
            retrieval(passage_embedding=passage_embedding,passage_embedding2id=passage_embedding2id,test_set=test_set,index=index,task_name=task_name)
        
        passage_embedding = None 
        del passage_embedding
        index.reset()
    else:
        logging.info("split search")
        _,passage_embedding2id=load_passage_embeddings(load_emb=False,emb_prefix='passage_')
        for test_set in test_sets:
            retrieval(passage_embedding=None,passage_embedding2id=passage_embedding2id,test_set=test_set,index=None,task_name=task_name)

if is_q2d_retrieval:
    trec_path = q2d_dev_q_retrieval(task_name=task_name)
    if format_to_MACRO_id:
        convert_trec_to_MARCO_id(path=trec_path,data_type=data_type,test_set=0,task_name=task_name,d2q_trec=False)    
if is_d2q_retrieval:
    dev_q_trec_path = d2q_from_dev_q_retrieval(task_name=task_name)
    train_q_trec_path = d2q_from_train_q_retrieval(task_name=task_name)
    if format_to_MACRO_id:
        convert_trec_to_MARCO_id(path=dev_q_trec_path,data_type=data_type,test_set=0,task_name=task_name,d2q_trec=True)
        convert_trec_to_MARCO_id(path=train_q_trec_path,data_type=data_type,test_set="training",task_name=task_name,d2q_trec=True)



