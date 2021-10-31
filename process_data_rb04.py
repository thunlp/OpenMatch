import argparse
import json
import os
from tqdm import tqdm
import itertools
import copy
import random

NUM_FOLDS = 5
SAMLING_RATES = [0.5, 0.2, 0.05, 0.02, 0.01, 0.002]
# NEG_DOCS_PER_Q = [5, 10, 20, 50, 100, 500]
NUM_Q = [99999, 50, 5, 1]
NEGS_MULTIPLE = [50, 10, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--id", type=str)
    args = parser.parse_args()

    collection_file = os.path.join(args.data_dir, args.id + "_doc.jsonl")
    qrels_file = os.path.join(args.data_dir, args.id + "_qrels")
    queries_file = os.path.join(args.data_dir, args.id + "_query.jsonl")
    run_file = os.path.join(args.data_dir, args.id + ".trec")

    print("Loading collection...")
    collection = {}
    with open(collection_file, "r") as f:
        for line in tqdm(f):
            obj = json.loads(line)
            collection[obj["docid"]] = obj

    print("Loading queries...")
    queries = {}
    with open(queries_file, "r") as f:
        for line in tqdm(f):
            obj = json.loads(line)
            queries[int(obj["qid"])] = obj

    print("Loading qrels...")
    qrels = {}
    removed = 0
    qrels_total_size = 0
    with open(qrels_file, "r") as f:
        for line in tqdm(f):
            qid, _, did, rel = line.strip().split()
            qid = int(qid)
            rel = int(rel)
            if rel < 0:
                rel = 0
            # assert did in collection
            if did not in collection:
                removed += 1
                continue
            if qid not in qrels:
                qrels[qid] = {}
                qrels[qid][did] = rel
            else:
                qrels[qid][did] = rel
            qrels_total_size += 1
    qrel_copy = copy.deepcopy(qrels)
    removed_q = 0
    for qid, doc_list in qrel_copy.items():
        num_pos = 0
        for docid, rel in doc_list.items():
            if rel > 0:
                num_pos += 1
        if num_pos <= 10:
            qrels_total_size -= len(qrels[qid])
            del qrels[qid]  # remove queries that do not contain any relevant docs
            removed_q += 1
    
    print("Removed {} documents which are not in collection from qrels.".format(removed))
    print("Removed {} queries that contain few relevant docs.".format(removed_q))
    print("Total {} judgements.".format(qrels_total_size))

    print("Loading trec...")
    trec = {}
    run_name = None
    with open(run_file, "r") as f:
        for line in tqdm(f):
            qid, _, did, rank, score, name = line.strip().split()
            run_name = name
            qid = int(qid)
            rank = int(rank)
            score = float(score)
            if did not in collection:
                continue
            if qid not in trec:
                # assert rank == 1, line
                trec[qid] = [{"docid": did, "rank": rank, "score": score}]
            else:
                # assert trec[qid][-1]["rank"] == rank - 1
                if len(trec[qid]) < 1000:
                    assert trec[qid][-1]["score"] >= score
                    trec[qid].append({"docid": did, "rank": rank, "score": score})


    print("Splitting data...")
    target_folders = []
    for i in range(NUM_FOLDS):
        target_folder = os.path.join(args.data_dir, "fold_" + str(i))
        target_folders.append(target_folder)
        if not os.path.exists(target_folder):
            print("Creating directory {}".format(target_folder))
            os.makedirs(target_folder)
        # few_shot_folder = os.path.join(target_folder, str(args.train_query_num) + "q")
        # if not os.path.exists(few_shot_folder):
        #     print("Creating directory {}".format(few_shot_folder))
        #     os.makedirs(few_shot_folder)
    # out_queries_files = [open(os.path.join(fd, args.id + "_query.jsonl"), "w") for fd in target_folders]
    # out_train_files = [os.path.join(fd, str(args.train_query_num) + "q", args.id + "_train_classification") for fd in target_folders]
    # out_train_files = [open("/dev/null", "w") for fd in target_folders]
    # out_test_files = [open(os.path.join(fd, args.id + "_test.jsonl"), "w") for fd in target_folders]
    # out_dev_files = [os.path.join(fd, str(args.train_query_num) + "q", args.id + "_dev") for fd in target_folders]
    # out_dev_files = [open("/dev/null", "w") for fd in target_folders]

    def dump_to_file(qrel_dict, file, repeat=1):
        with open(file, "w") as f:
            for qid, doc in qrel_dict.items():
                for did, rel in doc.items():
                    query = queries[qid]["query"]
                    # doc_text = collection[did]["title"] + " " + collection[did]["bodytext"]
                    for _ in range(repeat if rel > 0 else 1):
                        f.write(json.dumps(
                            {
                                "query": query,
                                "title": collection[did]["title"],
                                "doc": collection[did]["bodytext"],
                                "label": 1 if rel > 0 else 0,
                                "query_id": str(qid),
                                "doc_id": did
                            }
                        ) + "\n")

    for test_fold in range(NUM_FOLDS):
        print("Fold {}".format(test_fold))
        qrels_copy = {}
        qrels_total_size_fold = 0
        for qid, v in qrels.items():
            if qid % NUM_FOLDS != test_fold: 
                qrels_copy[qid] = copy.deepcopy(v)
                qrels_total_size_fold += len(v)
        cur_sampling = 0
        # cur_size = qrels_total_size_fold
        cur_size = len(qrels_copy)
        target_folder = target_folders[test_fold]
        for num_queries in NUM_Q:
            # removing queries until reaching num_queries
            print("--num_queries={}".format(num_queries))
            while cur_size > num_queries:
                qid_to_be_deleted = random.choice(list(qrels_copy.keys()))
                cur_size -= 1
                del qrels_copy[qid_to_be_deleted]
            assert cur_size <= num_queries
            # print("----", end="")
            target_qrel = {a:{x:{} for x in qrels_copy.keys()} for a in NEGS_MULTIPLE}
            for qid, rel_list in qrels_copy.items():
                # print("qid {} ".format(qid), end="")
                num_positives = 0
                num_negatives = 0
                pos_docs = set()
                neg_docs = set()
                for docid, rel in rel_list.items():
                    if rel > 0:
                        num_positives += 1
                        pos_docs.add(docid)
                        for neg_num, qrel in target_qrel.items():
                            qrel[qid][docid] = 1
                assert num_positives != 0
                for run_item in trec[qid]:
                    docid = run_item["docid"]
                    if docid not in pos_docs:  # neg doc
                        neg_docs.add(docid)
                        num_negatives += 1
                print("----qid {} pos_num {} neg_num {}".format(qid, num_positives, num_negatives))
                expected_num_negatives = [x*num_positives for x in NEGS_MULTIPLE]
                print("----expected_num_negatives {}".format(expected_num_negatives))
                for expn in expected_num_negatives:
                    while num_negatives > expn:
                        did_to_be_deleted = random.sample(neg_docs, 1)[0]
                        num_negatives -= 1
                        neg_docs.remove(did_to_be_deleted)
                    assert num_negatives <= expn
                    target_target_qrel = copy.deepcopy(target_qrel)
                    for did in neg_docs:
                        # target_target_qrel
                        target_qrel[expn//num_positives][qid][did] = 0
            for neg_num, qrel in target_qrel.items():
                dump_to_file(qrel, os.path.join(target_folder, "train.{}q.{}xneg.jsonl".format(num_queries, neg_num)), repeat=neg_num)
                


if __name__ == "__main__":
    main()