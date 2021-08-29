import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_dir", type=str, default="/data/private/yushi/meta_robust04/")
    parser.add_argument("--my_dir", type=str, default="/data/private/yushi/ReInfoSelect_Testing_Data/robust04")
    args = parser.parse_args()

    for fold in range(5):
        folder_name = "fold_" + str(fold)
        triples_file = os.path.join(args.meta_dir, folder_name, "triples.jsonl")
        docid_to_doc_file = os.path.join(args.meta_dir, folder_name, "docid2doc.jsonl")
        qid_to_query_file = os.path.join(args.meta_dir, folder_name, "qid2query.jsonl")
        output_triples_file = os.path.join(args.my_dir, folder_name, "sunsi_rb04_train_triples.jsonl")

        queries = {}
        with open(qid_to_query_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                queries[obj["qid"]] = obj["query"]
        
        docs = {}
        with open(docid_to_doc_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                docs[obj["docid"]] = obj["doc"]

        with open(triples_file, "r") as f, open(output_triples_file, "w") as g:
            for line in f:
                obj = json.loads(line)
                qid, doc_pos_id, doc_neg_id = obj["qid"], obj["pos_docid"], obj["neg_docid"]
                query = queries[qid]
                doc_pos = docs[doc_pos_id]
                doc_neg = docs[doc_neg_id]
                g.write(json.dumps(
                    {
                        "query": query,
                        "doc_pos": doc_pos,
                        "doc_neg": doc_neg
                    }
                ) + "\n")


if __name__ == "__main__":
    main()