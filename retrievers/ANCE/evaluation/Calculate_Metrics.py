#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# # Define params below

# In[2]:

task_name='FirstP_Pas_512'
checkpoint_path = "../data/raw_data/OSPass512/ann_data/"
checkpoint = 520000
data_type = 1 # 0 for document, 1 for passage
test_set = 0 # 0 for dev_set, 1 for eval_set
raw_data_dir = "../data/raw_data/"
processed_data_dir = "../data/raw_data/ann_data_roberta-base_512/"
trec_save_path = f"{task_name}-data-type-{data_type}_test-set-{test_set}_ckpt-{checkpoint}.trec"
print("start time:",time.asctime( time.localtime(time.time()) ))
# # Load Qrel

# In[3]:


if data_type == 0:
    topN = 100
else:
    topN = 1000
dev_query_positive_id = {}
query_positive_id_path = os.path.join(processed_data_dir, "dev-qrel.tsv")

with open(query_positive_id_path, 'r', encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [topicid, docid, rel] in tsvreader:
        topicid = int(topicid)
        docid = int(docid)
        if topicid not in dev_query_positive_id:
            dev_query_positive_id[topicid] = {}
        dev_query_positive_id[topicid][docid] = int(rel)


# # Prepare rerank data

# In[4]:


qidmap_path = processed_data_dir+"/qid2offset.pickle"
pidmap_path = processed_data_dir+"/pid2offset.pickle"
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

with open(pidmap_path, 'rb') as handle:
    pidmap = pickle.load(handle)

qset = set()
with gzip.open(query_path, 'rt', encoding='utf-8') if query_path[-2:] == "gz" else open(query_path, 'rt', encoding='utf-8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [qid, query] in tsvreader:
        qset.add(qid)

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


# # Calculate Metrics

# In[6]:


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
                trec_save_path,topN=10000):

    print(f"saving trec file to {trec_save_path}")
    output_f = open(trec_save_path,"w")
    # qids_to_ranked_candidate_passages = {} 

    for query_idx in range(len(I_nearest_neighbor)): 
        # seen_pid = set()
        query_id = query_embedding2id[query_idx]
        # prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        # if query_id in qids_to_ranked_candidate_passages:
        #     pass    
        # else:
        #     # By default, all PIDs in the list of 1000 are 0. Only override those that are given
        #     tmp = [0] * 1000
        #     qids_to_ranked_candidate_passages[query_id] = tmp
        for rank,idx in enumerate(selected_ann_idx):
            pred_pid = passage_embedding2id[idx]
            output_f.write(f"{str(query_id)} Q0 {str(pred_pid)} {str(rank+1)} {str(-D_nearest_neighbor[query_idx][rank])} ance\n")

            # if not pred_pid in seen_pid:
            #     # this check handles multiple vector per document
            #     qids_to_ranked_candidate_passages[query_id][rank]=pred_pid
            #     Atotal += 1
            #     if pred_pid not in dev_query_positive_id[query_id]:
            #         Alabeled += 1
            #     if rank < 10:
            #         total += 1
            #         if pred_pid not in dev_query_positive_id[query_id]:
            #             labeled += 1
            #     rank += 1
            #     prediction[query_id][pred_pid] = -rank
            #     seen_pid.add(pred_pid)

    output_f.close()

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


# In[7]:


dev_query_embedding = []
dev_query_embedding2id = []
passage_embedding = []
passage_embedding2id = []
for i in range(8):
    try:
        with open(checkpoint_path + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
            dev_query_embedding.append(pickle.load(handle))
        with open(checkpoint_path + "dev_query_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
            dev_query_embedding2id.append(pickle.load(handle))
        with open(checkpoint_path + "passage_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
            passage_embedding.append(pickle.load(handle))
        with open(checkpoint_path + "passage_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
            passage_embedding2id.append(pickle.load(handle))
    except:
        break
if (not dev_query_embedding) or (not dev_query_embedding2id) or (not passage_embedding) or not (passage_embedding2id):
    print("No data found for checkpoint: ",checkpoint)

dev_query_embedding = np.concatenate(dev_query_embedding, axis=0)
dev_query_embedding2id = np.concatenate(dev_query_embedding2id, axis=0)
passage_embedding = np.concatenate(passage_embedding, axis=0)
passage_embedding2id = np.concatenate(passage_embedding2id, axis=0)


# # reranking metrics

# In[8]:

print("read embeddings for faiss indexing:")

pidmap = collections.defaultdict(list)
for i in range(len(passage_embedding2id)):
    pidmap[passage_embedding2id[i]].append(i)  # abs pos(key) to rele pos(val)
    
rerank_data = {}
all_dev_I = []
for i,qid in tqdm(enumerate(dev_query_embedding2id)):
    p_set = []
    p_set_map = {}
    if qid not in bm25:
        print(qid,"not in bm25")
    else:
        count = 0
        for k,pid in enumerate(bm25[qid]):
            if pid in pidmap:
                for val in pidmap[pid]:
                    p_set.append(passage_embedding[val])
                    p_set_map[count] = val # new rele pos(key) to old rele pos(val)
                    count += 1
            else:
                print(pid,"not in passages")
    dim = passage_embedding.shape[1]
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(dim)
    p_set =  np.asarray(p_set)
    cpu_index.add(p_set)    
    _, dev_I = cpu_index.search(dev_query_embedding[i:i+1], len(p_set))
    for j in range(len(dev_I[0])):
        dev_I[0][j] = p_set_map[dev_I[0][j]]
    all_dev_I.append(dev_I[0])
result = EvalDevQuery(dev_query_embedding2id, passage_embedding2id, dev_query_positive_id, all_dev_I, topN)
final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, metrics, prediction = result
print("Reranking Results for checkpoint "+str(checkpoint))
print("Reranking NDCG@10:" + str(final_ndcg))
print("Reranking map@10:" + str(final_Map))
print("Reranking pytrec_mrr:" + str(final_mrr))
print("Reranking recall@"+str(topN)+":" + str(final_recall))
print("Reranking hole rate@10:" + str(hole_rate))
print("Reranking hole rate:" + str(Ahole_rate))
print("Reranking ms_mrr:" + str(ms_mrr))


# # full ranking metrics

# In[9]:


dim = passage_embedding.shape[1]
faiss.omp_set_num_threads(16)
cpu_index = faiss.IndexFlatIP(dim)
cpu_index.add(passage_embedding)    
dev_D, dev_I = cpu_index.search(dev_query_embedding, topN)

save_trec_file(dev_query_embedding2id, passage_embedding2id,
                dev_I,dev_D,trec_save_path)

result = EvalDevQuery(dev_query_embedding2id, passage_embedding2id, dev_query_positive_id, dev_I, topN)
final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, metrics, prediction = result
print("Results for checkpoint "+str(checkpoint))
print("NDCG@10:" + str(final_ndcg))
print("map@10:" + str(final_Map))
print("pytrec_mrr:" + str(final_mrr))
print("recall@"+str(topN)+":" + str(final_recall))
print("hole rate@10:" + str(hole_rate))
print("hole rate:" + str(Ahole_rate))
print("ms_mrr:" + str(ms_mrr))

print("end time:",time.asctime( time.localtime(time.time()) ))
# In[ ]:





