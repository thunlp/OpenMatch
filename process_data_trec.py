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
    print("Removed {} documents which are not in collection from qrels.".format(removed))
    print("Total {} judgements.".format(qrels_total_size))

    # print("Loading trec...")
    # trec = {}
    # run_name = None
    # with open(run_file, "r") as f:
    #     for line in tqdm(f):
    #         qid, _, did, rank, score, name = line.strip().split()
    #         run_name = name
    #         qid = int(qid)
    #         rank = int(rank)
    #         score = float(score)
    #         if did not in collection:
    #             continue
    #         if qid not in trec:
    #             # assert rank == 1, line
    #             trec[qid] = [{"docid": did, "rank": rank, "score": score}]
    #         else:
    #             # assert trec[qid][-1]["rank"] == rank - 1
    #             # if len(trec[qid]) < 100:
    #             assert trec[qid][-1]["score"] >= score
    #             trec[qid].append({"docid": did, "rank": rank, "score": score})


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

    def dump_to_file(qrel_dict, file):
        with open(file, "w") as f:
            for qid, doc in qrel_dict.items():
                for did, rel in doc.items():
                    query = queries[qid]["query"]
                    doc_text = collection[did]["title"] + " " + collection[did]["bodytext"]
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
        print("Fold {}. Deleting".format(test_fold), end=" ")
        qrels_copy = {}
        qrels_total_size_fold = 0
        for qid, v in qrels.items():
            if qid % NUM_FOLDS != test_fold: 
                qrels_copy[qid] = copy.deepcopy(v)
                qrels_total_size_fold += len(v)
        cur_sampling = 0
        cur_size = qrels_total_size_fold
        target_folder = target_folders[test_fold]
        while cur_sampling < len(SAMLING_RATES):
            qid_to_be_deleted = random.choice(list(qrels_copy.keys()))
            qrel_to_be_deleted = copy.deepcopy(qrels_copy[qid_to_be_deleted])
            cur_size -= len(qrel_to_be_deleted)
            del qrels_copy[qid_to_be_deleted]  # random deletion
            print(qid_to_be_deleted, end=" ")
            cur_judgment_limit = int(SAMLING_RATES[cur_sampling] * qrels_total_size_fold)
            if cur_size <= cur_judgment_limit:
                if cur_size < cur_judgment_limit:
                    print("\nSampling rate lower than {}. Re-inserting {}.".format(SAMLING_RATES[cur_sampling], qid_to_be_deleted))
                    # re-insert deleted query
                    qrels_copy[qid_to_be_deleted] = qrel_to_be_deleted
                    cur_size += len(qrel_to_be_deleted)
                    assert cur_size >= cur_judgment_limit
                    print("Current judgment size {}. Deleting judgements until the number drops to {}.".format(cur_size, cur_judgment_limit))
                    # Delete judgments
                    while cur_size > cur_judgment_limit:
                        delete_from = random.choice(list(qrels_copy.keys()))
                        del qrels_copy[delete_from][random.choice(list(qrels_copy[delete_from].keys()))]
                        if qrels_copy[delete_from] == {}:
                            del qrels_copy[delete_from]
                        cur_size -= 1

                filename = os.path.join(target_folder, "{}_train_classification_sample_{}.jsonl".format(args.id, SAMLING_RATES[cur_sampling]))
                dump_to_file(qrels_copy, filename)
                cur_sampling += 1
                


if __name__ == "__main__":
    main()