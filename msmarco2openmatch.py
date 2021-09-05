
#src_path  :/data/private/yushi/collections/msmarco-passage
#des_path  :msmarco
#src 中文件说明 collection.tsv ,
# dev {qrels.dev.small.tsv,queries.dev.small.tsv}   
# train{ qrels.train.tsv,queries.train.tsv}
#1. collection   [docid,doc]
#2. qrels [qid,_,did,r/ir]
#3. querys [qid,query]
#target 
#[query,doc,label,query_id,doc_id]

#把msmarco里的文件全部转换为openmatch可以接受的
#######用来构造给定query数目，带有不同负例的多个数据集
#######例子中是5个query,负例数分别为5,10,20,40,80


import random
import pandas as pd
import os
import json
neg_num=[5,10,20,40,80]
src_path=os.path.join('/data',"private","yushi","collections","msmarco-passage")
collection_file=os.path.join(src_path,"collection.tsv")
qrels_dev_small_file=os.path.join(src_path,"qrels.dev.small.tsv")
qrels_train_file=os.path.join(src_path,"qrels.train.tsv")
queries_dev_small_file=os.path.join(src_path,"queries.dev.small.tsv")
queries_train_file=os.path.join(src_path,"queries.train.tsv")
trec_train_file=os.path.join('/data/private/yushi/runs/run.msmarco-passage.train.bm25tuned.txt')
trec_dev_file=os.path.join('/data/private/yushi/runs/run.msmarco-passage.bm25tuned.txt')


collection=pd.read_csv(collection_file,delimiter='\t',names=['did','doc'])
qrels_dev_small=pd.read_csv(qrels_dev_small_file,delimiter='\t',names=['qid','_','did','rel'])
qrels_train=pd.read_csv(qrels_train_file,delimiter='\t',names=['qid','_','did','rel'])
queries_dev_small=pd.read_csv(queries_dev_small_file,delimiter='\t',names=['qid','query'])
queries_train=pd.read_csv(queries_train_file,delimiter='\t',names=['qid','query'])
trec_train=pd.read_csv(trec_train_file,delimiter=' ',names=['qid','_','did','rank','score','tools'])
trec_dev=pd.read_csv(trec_dev_file,delimiter=' ',names=['qid','_','did','rank','score','tools'])

#构建数据集单元
def build_dataset(
    qid_list,#qid的list
    queries,#
    collection,#
    trec,#检索列表
    qrels,#ground_truth
    target_list,#所需要的构造的数据集
    target_did_list,#target_list里的
    target_num,#所需要的构造的数据集中每个query带有的例子的数目，可能是正例的数目也可能是反例的数目
    pos,#代表是不是构造正例
    cur
    ):
    for qid in qid_list:
            cur_num=cur#目前已经有的正例数目
            trec_qid=trec[trec['qid']==qid]
            trec_did=[trec_qid.iloc[i]['did'] for i in range(len(trec_qid))]#所有被检索过的doc_id
            qrels_qid=qrels[qrels['qid']==qid]
            rel_did=[qrels_qid.iloc[i]['did'] for i in range(len(qrels_qid))]#所有相关的doc_id

            for did in trec_did:
                if cur_num==target_num:
                    break
                if did >= len(doc_list):
                    continue
                if did in rel_did and pos==True and did not in target_did_list:
                    if cur_num<target_num:
                        label=1
                        target_did_list.append(did)
                        doc=collection.iloc[did]['doc']
                        query=queries[queries['qid']==qid].iloc[0]['query']
                        target_list.append(
                            {'query':query,'doc':doc,'label':label,'query_id':int(qid),'doc_id':int(did)}
                        )
                        cur_num+=1
                elif did not in rel_did and pos==False and did not in target_did_list:
                    if cur_num<target_num:
                        label=0
                        target_did_list.append(did)
                        doc=collection.iloc[did]['doc']
                        query=queries[queries['qid']==qid].iloc[0]['query']
                        target_list.append(
                            {'query':query,'doc':doc,'label':label,'query_id':int(qid),'doc_id':int(did)}
                        )
                        cur_num+=1
    return target_list,target_did_list


#构建dev集合
doc_list=list(collection['did'])#所有候选的doc集合
qid_dev_small_list=list(queries_dev_small['qid'])#所有dev集合query的qid集合
dev_list=[]#dev集合
for qid in qid_dev_small_list:
    trec_dev_qid=trec_dev[trec_dev['qid']==qid]
    trec_did=[trec_dev_qid.iloc[i]['did'] for i in range(len(trec_dev_qid))]#所有被检索过的doc_id
    qrels_dev_qid=qrels_dev_small[qrels_dev_small['qid']==qid]
    rel_did=[qrels_dev_qid.iloc[i]['did'] for i in range(len(qrels_dev_qid))]#所有相关的doc_id
    for did in trec_did:
        if did >= len(doc_list):
            continue
        if did in rel_did:
            label=1
        else:
            label=0
        doc=collection.iloc[did]['doc']
        query=queries_dev_small[queries_dev_small['qid']==qid].iloc[0]['query']
        score=trec_dev_qid[trec_dev_qid['did']==did].iloc[0]['score']
        dev_list.append(
            {'query':query,'doc':doc,'label':label,'query_id':int(qid),'doc_id':int(did),'retrieval_score':float(score)}
        )
#将dev集数目变为原来的1/10
length=len(dev_list)
dev_list=random.sample(dev_list,int(length/10))
for item in dev_list:
    with open("msmarco_dev.jsonl","a+") as f:
        json.dump(item,f)
        f.write('\n')
#构建train集合
#filename="msmarco_train_classification_neg{}.jsonl".format(neg)
qid_train_list=list(queries_train['qid'])#所有train集合query的qid集合
while True:
    print(1)
    #train_list=[]#train集合
    #打乱出一组qid
    sample_qid_list=random.sample(qid_train_list,5)
    #首先为这组构造正例
    train_pos_list=[],train_pos_did_list=[]
    train_pos_list,_=build_dataset(
        sample_qid_list,queries_train,collection,trec_train,qrels_train,train_pos_list,train_pos_did_list,1,True,0
        )
    if len(train_pos_list)!=5:
        continue
    #train_pos_list里面已经有5个正例
    train_neg5_list=[]
    train_neg5_did_list=[]
    train_neg10_list=[]
    train_neg10_did_list=[]
    train_neg20_list=[]
    train_neg20_did_list=[]
    train_neg40_list=[]
    train_neg40_did_list=[]
    train_neg80_list=[]
    train_neg80_did_list=[]
    #首先构造五个负例的list
    train_neg5_list,train_neg5_did_list=build_dataset(
        sample_qid_list,queries_train,collection,trec_train,qrels_train,train_neg5_list,train_neg5_did_list,5,False,0
        )
    if len(train_neg5_list)!=25:
        continue
    #构造十个负例的list
    for item in train_neg5_list:
        train_neg10_list.append(item)
        train_neg10_did_list.append(item['doc_id'])
    train_neg10_list,train_neg10_did_list=build_dataset(
        sample_qid_list,queries_train,collection,trec_train,qrels_train,train_neg10_list,train_neg10_did_list,10,False,5
        )
    if len(train_neg10_list)!=50:
        continue
    #构造二十个负例的list
    for item in train_neg10_list:
        train_neg20_list.append(item)
        train_neg20_did_list.append(item['doc_id'])
    train_neg20_list,train_ne20_did_list=build_dataset(
        sample_qid_list,queries_train,collection,trec_train,qrels_train,train_neg20_list,train_neg20_did_list,20,False,10
        )
    if len(train_neg20_list)!=100:
        continue    
    #构造四十个负例的list
    for item in train_neg20_list:
        train_neg40_list.append(item)
        train_neg40_did_list.append(item['doc_id'])
    train_neg40_list,train_neg40_did_list=build_dataset(
        sample_qid_list,queries_train,collection,trec_train,qrels_train,train_neg40_list,train_neg40_did_list,40,False,20
        )
    if len(train_neg40_list)!=200:
        continue
    #构造八十个负例的list
    for item in train_neg40_list:
        train_neg80_list.append(item)
        train_neg80_did_list.append(item['doc_id'])
    train_neg80_list,train_neg80_did_list=build_dataset(
        sample_qid_list,queries_train,collection,trec_train,qrels_train,train_neg40_list,train_neg40_did_list,80,False,40
        )
    if len(train_neg80_list)!=400:
        continue  
    train_5_list=train_pos_list+train_neg5_list
    train_10_list=train_pos_list+train_neg10_list
    train_20_list=train_pos_list+train_neg20_list
    train_40_list=train_pos_list+train_neg40_list
    train_80_list=train_pos_list+train_neg80_list
    for item in train_5_list:
        with open("msmarco_train_classification_neg5.jsonl","a+") as f:
            json.dump(item,f)
            f.write('\n')
    for item in train_10_list:
        with open("msmarco_train_classification_neg10.jsonl","a+") as f:
            json.dump(item,f)
            f.write('\n')
    for item in train_20_list:
        with open("msmarco_train_classification_neg20.jsonl","a+") as f:
            json.dump(item,f)
            f.write('\n')
    for item in train_40_list:
        with open("msmarco_train_classification_neg40.jsonl","a+") as f:
            json.dump(item,f)
            f.write('\n')
    for item in train_80_list:
        with open("msmarco_train_classification_neg80.jsonl","a+") as f:
            json.dump(item,f)
            f.write('\n')
    break
