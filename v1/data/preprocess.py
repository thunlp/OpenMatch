import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_trec', type=str)
    parser.add_argument('-input_qrels', type=str, default=None)
    parser.add_argument('-input_queries', type=str)
    parser.add_argument('-input_docs', type=str)
    parser.add_argument('-output', type=str)
    args = parser.parse_args()

    qs = {}
    if args.input_queries.split('.')[-1] == 'json' or args.input_queries.split('.')[-1] == 'jsonl':
        with open(args.input_queries, 'r') as r:
            for line in r:
                line = json.loads(line)
                qs[line['query_id']] = line['query']
    else:
        with open(args.input_queries, 'r') as r:
            for line in r:
                line = line.strip('\n').split('\t')
                qs[line[0]] = line[1]

    ds = {}
    if args.input_queries.split('.')[-1] == 'json' or args.input_queries.split('.')[-1] == 'jsonl':
        with open(args.input_docs, 'r') as r:
            for line in r:
                line = json.loads(line)
                ds[line['doc_id']] = line['doc'].strip()
    else:
        with open(args.input_docs, 'r') as r:
            for line in r:
                line = line.strip('\n').split('\t')
                if len(line) > 2:
                    ds[line[0]] = line[-2] + ' [SEP] ' + line[-1]
                else:
                    ds[line[0]] = line[1]

    if args.input_qrels is not None:
        qpls = {}
        with open(args.input_qrels, 'r') as r:
            for line in r:
                line = line.strip().split()
                if line[0] not in qpls:
                    qpls[line[0]] = {}
                qpls[line[0]][line[2]] = int(line[3])

    f = open(args.output, 'w')
    with open(args.input_trec, 'r') as r:
        for line in r:
            line = line.strip().split()
            if line[0] not in qs or line[2] not in ds:
                continue
            if args.input_qrels is not None:
                if line[0] in qpls and line[2] in qpls[line[0]]:
                    label = qpls[line[0]][line[2]]
                else:
                    label = 0
                f.write(json.dumps({'query': qs[line[0]], 'doc': ds[line[2]], 'label': label, 'query_id': line[0], 'doc_id': line[2], 'retrieval_score': float(line[4])}) + '\n')
            else:
                f.write(json.dumps({'query': qs[line[0]], 'doc': ds[line[2]], 'label': -1, 'query_id': line[0], 'doc_id': line[2], 'retrieval_score': float(line[4])}) + '\n')
    f.close()

if __name__ == "__main__":
    main()
