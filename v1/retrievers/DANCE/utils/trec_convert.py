import os
import pickle
import numpy as np
from tqdm import tqdm

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

        for rank,idx in enumerate(selected_ann_idx):
            pred_pid = passage_embedding2id[idx]
            output_f.write(f"{str(query_id)} Q0 {str(pred_pid)} {str(rank+1)} {str(D_nearest_neighbor[query_idx][rank])} ance\n")

def convert_trec_to_MARCO_id(data_type,test_set,processed_data_dir,trec_path,d2q_reversed_trec_file=False):

    if test_set=="training":
        with open(os.path.join(processed_data_dir,'train-query_qid2offset.pickle'),'rb') as f:
            qid2offset = pickle.load(f)
    elif test_set=="docleaderboard":
        with open(os.path.join(processed_data_dir,'docleaderboard_qid2offset.pickle'),'rb') as f:
            qid2offset = pickle.load(f)
    elif test_set=="trec2019":
        with open(os.path.join(processed_data_dir,'dev-query_qid2offset.pickle'),'rb') as f:
            qid2offset = pickle.load(f)
    elif test_set=="marcodev":
        with open(os.path.join(processed_data_dir,'real-dev-query_qid2offset.pickle'),'rb') as f:
            qid2offset = pickle.load(f)
    else:
        logging.error("wrong test type")
        exit()
    offset2qid = {}
    for k in qid2offset:
        offset2qid[qid2offset[k]]=k

    with open(os.path.join(processed_data_dir,'pid2offset.pickle'),'rb') as f:
        pid2offset = pickle.load(f)
    offset2pid = {}
    for k in pid2offset:
        offset2pid[pid2offset[k]]=k

    # for path in trec_save_path_list:
    print("processing and formatting file:",trec_path)
    with open(trec_path) as f:
        lines=f.readlines()
    with open(trec_path.replace(".trec",".formatted.trec"),"w") as f:
        for line in tqdm(lines):
            if d2q_reversed_trec_file:
                pid , Q0, qid, rank, score, tag = line.strip().split(' ')
            else:
                qid , Q0, pid, rank, score, tag = line.strip().split(' ')
            # print(offset2qid[int(qid)] , Q0, pid, rank, score.replace('-',''), tag)
            if d2q_reversed_trec_file:
                if data_type==0:
                    f.write(f"D{offset2pid[int(pid)]} {Q0} {offset2qid[int(qid)]} {rank} {score.replace('-','')} {tag}\n")
                else:
                    f.write(f"{offset2pid[int(pid)]} {Q0} {offset2qid[int(qid)]} {rank} {score.replace('-','')} {tag}\n")
            else:
                if data_type==0:
                    f.write(f"{offset2qid[int(qid)]} {Q0} D{offset2pid[int(pid)]} {rank} {score.replace('-','')} {tag}\n")
                else:
                    f.write(f"{offset2qid[int(qid)]} {Q0} {offset2pid[int(pid)]} {rank} {score.replace('-','')} {tag}\n")