import glob
import pickle
import os
import tqdm

data_type=1
test_set=0

processed_data_dir = "../data/raw_data/ann_data_roberta-base_512/"
trec_save_path = glob.glob(f"data-type-{data_type}_test-set-{test_set}_ckpt-*.trec")

with open(os.path.join(processed_data_dir,'qid2offset.pickle'),'rb') as f:
    qid2offset = pickle.load(f)
offset2qid = {}
for k in qid2offset:
    offset2qid[qid2offset[k]]=k

with open(os.path.join(processed_data_dir,'pid2offset.pickle'),'rb') as f:
    pid2offset = pickle.load(f)
offset2pid = {}
for k in pid2offset:
    offset2pid[pid2offset[k]]=k


#for k in offset2qid:
#    print(k,offset2qid[k])

for path in tqdm.tqdm(trec_save_path):
    with open(path) as f:
        lines=f.readlines()
    with open(path.replace(".trec",".formatted.trec"),"w") as f:
        for line in lines:
            qid , Q0, pid, rank, score, tag = line.strip().split(' ')
            # print(offset2qid[int(qid)] , Q0, pid, rank, score.replace('-',''), tag)
            if data_type==0:
                f.write(f"{offset2qid[int(qid)]} {Q0} D{offset2pid[int(pid)]} {rank} {score.replace('-','')} {tag}\n")
            else:
                f.write(f"{offset2qid[int(qid)]} {Q0} {offset2pid[int(pid)]} {rank} {score.replace('-','')} {tag}\n")
#    break
