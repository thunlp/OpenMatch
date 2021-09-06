import json
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--om_run", type=str)
parser.add_argument("--out_name", type=str)
args = parser.parse_args()

rank = 0
last_qid = None
with open(args.om_run, "r") as f, open(args.out_name + ".trec", "w") as g, open(args.out_name + ".shuf.trec", "w") as h:
    for line in f:
        obj = json.loads(line)
        if obj["query_id"] != last_qid:
            last_qid = obj["query_id"]
            rank = 1
        g.write(f"{obj['query_id']} Q0 {obj['doc_id']} {rank} {obj['retrieval_score']} first_stage\n")
        h.write(f"{obj['query_id']} Q0 {obj['doc_id']} {rank} {random.random()} first_stage_rand\n")
        rank += 1