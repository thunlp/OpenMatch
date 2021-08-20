import random
import pickle
import sys

def dev_txt_file_split(query_file,qrel_file,split_num):
    assert split_num > 1
    
    query_lines={}
    qrel_lines={}
    qids=[]
    with open(query_file,'r') as f:
        for line in f:
            qid,_=line.strip().split('\t')
            qid=int(qid)
            query_lines[qid]=line
            qids.append(qid)
    with open(qrel_file,'r') as f:
        for line in f:
            qid,_,_,_ = line.strip().split(' ')
            qid=int(qid)
            qrel_lines[qid] = line
    
    random.shuffle(qids)

    factor = int(len(qids)/split_num)
    s=0
    for i in range(split_num):
        
        if i == split_num-1:
            t = len(qids)
        else:
            t= s + factor

        with open(query_file + f".cross_validation.{i}", 'w') as qf, open(qrel_file + f".cross_validation.{i}", 'w') as rf:
            for qid in qids[s:t]:
                qf.write(query_lines[qid])
                rf.write(qrel_lines[qid])
        s=t

def generate_split_file(qrel_file,split_num):

    assert split_num > 1

    reversed_qrel = qrel_file.replace('.tsv','-d2q-reversed.tsv')

    qids=[]
    qid2pid={}
    with open(qrel_file,'r') as f, open(reversed_qrel,'w') as out_f:
        for line in f:
            qid,_,pid,_ = line.strip().split('\t')
            # query_lines[qid]=line
            qids.append(qid)
            qid2pid[qid]=pid
            out_f.write(f"{pid}\t0\t{qid}\t1\n")

    random.shuffle(qids)

    factor = int(len(qids)/split_num)
    split={}
    reversed_split={}
    s=0
    for i in range(split_num):
        
        if i == split_num-1:
            t = len(qids)
        else:
            t= s + factor
        
        split[i]=[ str(qid) for qid in qids[s:t]]
        reversed_split[i]=[ str(qid2pid[qid]) for qid in qids[s:t]]
        s=t

    with open(qrel_file+f"{split_num}_fold.split_dict", 'wb') as f:
        pickle.dump(split,f)

    with open(reversed_qrel+f"{split_num}_fold.split_dict", 'wb') as f:
        pickle.dump(reversed_split,f)
    
    


def main():
    generate_split_file(sys.argv[1],int(sys.argv[2]))
    
if __name__ == '__main__':
    main()
