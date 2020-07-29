import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_qrels', type=str)
    parser.add_argument('-input_trec', type=str)
    parser.add_argument('-output_topk', type=int, default=1000)
    parser.add_argument('-output_trec', type=str)
    args = parser.parse_args()

    docs = {}
    with open(args.input_qrels, 'r') as r:
        for line in r:
            line = line.strip().split()
            docs[line[2]] = 1

    f = open(args.output_trec, 'w')
    qds = {}
    with open(args.input_trec, 'r') as r:
        for line in r:
            line = line.strip().split()
            if line[0] not in qds:
                qds[line[0]] = []
            if len(qds[line[0]]) >= args.output_topk:
                continue
            if line[2] not in docs:
                qds[line[0]].append(line[2])
                f.write(' '.join(line) + '\n')
    f.close()

if __name__ == "__main__":
    main()
