import argparse
import json
import os
from tqdm import tqdm
import itertools
import copy
import random

NUM_FOLDS = 5
NEG_DOCS_PER_Q = [5, 10, 20, 50, 100, 500]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_query_num", type=int)
    parser.add_argument("--repeat", type=int, default=5)
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
    print("Removed {} documents which are not in collection from qrels.".format(removed))

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
                if len(trec[qid]) < 100:
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
    # out_train_files = [open(os.path.join(fd, args.id + "_train_classification.jsonl"), "w") for fd in target_folders]
    out_train_files = [open("/dev/null", "w") for fd in target_folders]
    # out_test_files = [open(os.path.join(fd, args.id + "_test.jsonl"), "w") for fd in target_folders]
    out_dev_files = [open(os.path.join(fd, args.id + "_dev.jsonl"), "w") for fd in target_folders]
    # out_dev_files = [open("/dev/null", "w") for fd in target_folders]


    # for fold_id in range(NUM_FOLDS):
    #     dev_qids = set()
    #     train_qids = set()
    #     test_fold_remainder = fold_id
    #     dev_fold_remainder = (test_fold_remainder + 1) % NUM_FOLDS

    #     for qid, rel_docs in qrels.items():
    #         # sorted_rel_docs = sorted(list(rel_docs.items()), key=lambda x:x[1], reverse=True)
    #         if qid % NUM_FOLDS == test_fold_remainder:
    #             doc_list = trec[qid]
    #             for i, entry in enumerate(doc_list):
    #                 out_test_files[fold_id].write(json.dumps(
    #                 {
    #                     "query": queries[qid], 
    #                     "doc": collection[entry["docid"]]["title"] + " " + collection[entry["docid"]]["bodytext"], 
    #                     "label": rel_docs[entry["docid"]] if entry["docid"] in rel_docs else 0, 
    #                     "query_id": str(qid),
    #                     "doc_id": entry["docid"],
    #                     "retrieval_score": entry["score"]
    #                 }) + "\n")
    #                 if i + 1 == 100:
    #                     break
    #         elif qid % NUM_FOLDS == dev_fold_remainder:
    #             dev_qids.add(qid)
    #         else:
    #             train_qids.add(qid)

    #     for num_negs in NEG_DOCS_PER_Q:
    #         for rep in range(args.repeat):
    #             target_dev_file = open(out_dev_files[fold_id] + ".select{}.neg{}.jsonl".format(rep, num_negs), "w")
    #             sampled_dev_qids = None
    #             while True:
    #                 resample = 0
    #                 sampled_dev_qids = random.sample(dev_qids, k=args.train_query_num)
    #                 for qid in sampled_dev_qids:
    #                     relevant_docs = set(qrels[qid].keys())
    #                     retrieved_docs = set([entry["docid"] for entry in trec[qid]])
    #                     irrelevant_docs = retrieved_docs - relevant_docs
    #                     if len(relevant_docs) >= 5 and len(irrelevant_docs) > num_negs:
    #                         resample += 1
    #                 if resample == len(sampled_dev_qids):
    #                     break
    #             for qid in sampled_dev_qids:
    #                 sorted_rel_docs = sorted(list(qrels[qid].items()), key=lambda x:x[1], reverse=True)
    #                 pos_docs = sorted_rel_docs[:5]  # keep top 5 relevant
    #                 for pos in pos_docs:
    #                     did, rel = pos
    #                     target_dev_file.write(json.dumps(
    #                     {
    #                         "query": queries[qid]["query"], 
    #                         "doc": collection[did]["title"] + " " + collection[did]["bodytext"], 
    #                         "label": qrels[qid][did] if did in qrels[qid] else 0, 
    #                         "query_id": str(qid),
    #                         "doc_id": did,
    #                         "retrieval_score": 0.0
    #                     }) + "\n")
    #                 doc_list = trec[qid]
    #                 # for i, entry in enumerate(doc_list):
    #                 neg_sample_range = 1000
    #                 if num_negs < 100:
    #                     neg_sample_range = 200
    #                 doc_list = doc_list[:neg_sample_range]
    #                 filtered_doc_list = []
    #                 for entry in doc_list:
    #                     if entry["docid"] not in qrels[qid]:  # negative
    #                         filtered_doc_list.append(entry)
    #                 # print(qid, len(filtered_doc_list))
    #                 sampled_neg_doc_list = random.sample(filtered_doc_list, k=num_negs)
    #                 for entry in sampled_neg_doc_list:
    #                     target_dev_file.write(json.dumps(
    #                     {
    #                         "query": queries[qid]["query"], 
    #                         "doc": collection[entry["docid"]]["title"] + " " + collection[entry["docid"]]["bodytext"], 
    #                         "label": 0, 
    #                         "query_id": str(qid),
    #                         "doc_id": entry["docid"],
    #                         "retrieval_score": 0.0
    #                     }) + "\n")
    #             target_dev_file.close()

    #             target_train_file = open(out_train_files[fold_id] + ".select{}.neg{}.jsonl".format(rep, num_negs), "w")
    #             sampled_train_qids = random.sample(train_qids, k=args.train_query_num)
    #             while True:
    #                 resample = 0
    #                 sampled_train_qids = random.sample(dev_qids, k=args.train_query_num)
    #                 for qid in sampled_train_qids:
    #                     relevant_docs = set(qrels[qid].keys())
    #                     retrieved_docs = set([entry["docid"] for entry in trec[qid]])
    #                     irrelevant_docs = retrieved_docs - relevant_docs
    #                     if len(relevant_docs) >= 5 and len(irrelevant_docs) > num_negs:
    #                         resample += 1
    #                 if resample == len(sampled_train_qids):
    #                     break
    #             for qid in sampled_train_qids:
    #                 sorted_rel_docs = sorted(list(qrels[qid].items()), key=lambda x:x[1], reverse=True)
    #                 pos_docs = sorted_rel_docs[:5]
    #                 for pos in pos_docs:
    #                     did, rel = pos
    #                     target_train_file.write(json.dumps(
    #                     {
    #                         "query": queries[qid]["query"], 
    #                         "doc": collection[did]["title"] + " " + collection[did]["bodytext"], 
    #                         "label": 1, 
    #                         "query_id": str(qid),
    #                         "doc_id": did,
    #                     }) + "\n")
    #                 doc_list = trec[qid]
    #                 # for i, entry in enumerate(doc_list):
    #                 neg_sample_range = 1000
    #                 # if num_negs < 100:
    #                 #     neg_sample_range = 200
    #                 doc_list = doc_list[:neg_sample_range]
    #                 filtered_doc_list = []
    #                 for entry in doc_list:
    #                     if entry["docid"] not in qrels[qid]:  # negative
    #                         filtered_doc_list.append(entry)
    #                 # print(qid, len(filtered_doc_list))
    #                 sampled_neg_doc_list = random.sample(filtered_doc_list, k=num_negs)
    #                 for entry in sampled_neg_doc_list:
    #                     target_train_file.write(json.dumps(
    #                     {
    #                         "query": queries[qid]["query"], 
    #                         "doc": collection[entry["docid"]]["title"] + " " + collection[entry["docid"]]["bodytext"], 
    #                         "label": 0, 
    #                         "query_id": str(qid),
    #                         "doc_id": entry["docid"],
    #                     }) + "\n")
    #             target_train_file.close()


    avg_pos_docs, avg_neg_docs = 0, 0
    for qid, query in tqdm(queries.items()):
        # print(qid, query)
        query = query["query"]
        test_fold = qid % NUM_FOLDS
        dev_fold = (test_fold + 1) % NUM_FOLDS
        if qid not in qrels:
            print("{} not found in qrels.".format(qid))
            continue
        qrel = qrels[qid]
        doc_list = trec[qid]
        # out_queries_files[fold].write(json.dumps(query) + "\n")
        # pos_docs = set([did for did, rel in qrel.items() if rel > 2])
        # print(len(pos_docs))
        # assert len(pos_docs) != 0
        # avg_pos_docs += len(pos_docs)
        # neg_docs = set([entry["docid"] for entry in doc_list if entry["docid"] not in pos_docs])
        # print(len(neg_docs))
        # assert len(neg_docs) != 0
        # avg_neg_docs += len(neg_docs)
        # pairs = list(itertools.product(pos_docs, neg_docs))

        # seen_docs = set([entry["docid"] for entry in doc_list])
        # qrel_copy = copy.deepcopy(qrel)
        # for entry in doc_list:
        #     did = entry["docid"]
        #     if did not in qrel:
        #         qrel_copy[did] = 0
        # for did, rel in qrel.items():
        #     if did not in seen_docs and rel <= 0:
        #         del qrel_copy[did]
        # print(qrel_copy)
        # pairs = []
        # for rel_level in range(4, 0, -1):
        #     for did, rel in qrel_copy.items():
        #         if rel == rel_level:
        #             for did2, rel2 in qrel_copy.items():
        #                 if rel2 == 0:  # strong contrast
        #                     pairs.append((did, did2))

        # max_rel = 0
        # max_rel_did = None
        # for did, rel in qrel_copy.items():
        #     if rel > max_rel:
        #         max_rel_did = did
        #         max_rel = rel
        # assert max_rel > 0
        # zero_rel_did = None
        # for did, rel in qrel_copy.items():
        #     if rel == 0:
        #         zero_rel_did = did
        #         break
        # assert zero_rel_did is not None

        pos_doc_ids = set()
        neg_doc_ids = set()
        for did, rel in qrel.items():
            if rel > 0:
                pos_doc_ids.add(did)
            elif rel == 0:
                neg_doc_ids.add(did)

        # tmp_size = len(pairs)
        # # if tmp_size < 200:
        # force_break = False
        # for rel_level in range(4, 0, -1):
        #     for did, rel in qrel_copy.items():
        #         if rel == rel_level:
        #             for did2, rel2 in qrel_copy.items():
        #                 if rel2 == rel_level - 1:  # weak contrast
        #                     pairs.append((did, did2))
        #                     tmp_size += 1
            #                 if tmp_size >= 200:
            #                     force_break = True
            #                     break
            #     if force_break:
            #         break
            # if force_break:
            #     break
                                
        # pairs = random.choices(pairs, k=200) if len(pairs) > 200 else pairs
        # random.shuffle(pairs)
        # assert len(pairs) > 0, (qid, qrel_copy, qrel)
        # print(len(pairs))
        # exit(0)
        # for i in range(NUM_FOLDS):
        #     if i == fold:
        #         continue
        #     for pair in pairs:
        #         out_train_files[i].write(json.dumps({"query": query, "doc_pos": collection[pair[0]]["title"] + " " + collection[pair[0]]["bodytext"], "doc_neg": collection[pair[1]]["title"] + " " + collection[pair[1]]["bodytext"]}) + "\n")
        for i in range(NUM_FOLDS):
            if i == test_fold:
                continue
            for did in pos_doc_ids:
                out_train_files[i].write(json.dumps(
                    {
                        "query": query,
                        "title": collection[did]["title"],
                        "doc": collection[did]["bodytext"],
                        "query_id": qid,
                        "doc_id": did,
                        "label": 1
                    }
                ) + "\n")
            for did in neg_doc_ids:
                out_train_files[i].write(json.dumps(
                    {
                        "query": query,
                        "title": collection[did]["title"],
                        "doc": collection[did]["bodytext"],
                        "query_id": qid,
                        "doc_id": did,
                        "label": 0
                    }
                ) + "\n")
        for entry in doc_list:
            out_dev_files[test_fold].write(json.dumps(
            {
                "query": query,
                "title": collection[entry["docid"]]["title"],
                "doc": collection[entry["docid"]]["bodytext"], 
                "label": qrel[entry["docid"]] if entry["docid"] in qrel else 0, 
                "query_id": str(qid),
                "doc_id": entry["docid"],
                "retrieval_score": entry["score"]
            }) + "\n")
    avg_pos_docs /= len(queries)
    avg_neg_docs /= len(queries)
    print("Total {} queries. Avg positive docs per q: {}, avg negative docs per q: {}.".format(len(queries), avg_pos_docs, avg_neg_docs))
    for f in out_dev_files:
        f.close()
        

if __name__ == "__main__":
    main()