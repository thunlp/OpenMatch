import argparse
import json
import csv
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trec_file", type=str)
    parser.add_argument("--kilt_queries_file", type=str, default=None)
    parser.add_argument("--passage_collection", type=str)
    parser.add_argument("--output_provenance_file", type=str)
    args = parser.parse_args()

    queries = []
    if args.kilt_queries_file is not None:
        with open(args.kilt_queries_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                queries.append(obj)

    pid2content = []
    with open(args.passage_collection, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        i = 0
        for row in tqdm(reader):
            pid, text, wikipedia_title, wikipedia_id, _, _ = row
            pid = int(pid)
            assert pid == i
            pid2content.append({"text": text, "wikipedia_title": wikipedia_title, "wikipedia_id": wikipedia_id})
            i += 1

    provenance = {}
    with open(args.trec_file, "r") as f:
        last_qid = 0
        for line in f:
            qid, _, pid, rank, score, _ = line.strip().split()
            qid = int(qid)
            pid = int(pid)
            rank = int(rank)
            real_qid = queries[qid - 1]["id"] if len(queries) > 0 else str(qid)
            if qid != last_qid:  # new query
                provenance[real_qid] = []
                last_qid = qid
            provenance[real_qid].append({"score": score, "text": pid2content[pid]["text"], "wikipedia_title": pid2content[pid]["wikipedia_title"], "wikipedia_id": pid2content[pid]["wikipedia_id"]})
    
    with open(args.output_provenance_file, "w") as f:
        json.dump(provenance, f, indent=4)