import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_qrels', type=str, default=None)
    parser.add_argument('-input_trec', type=str)
    parser.add_argument('-topk', type=int, default=1000)
    parser.add_argument('-output', type=str)
    args = parser.parse_args()

    last_qds = {}
    if args.input_qrels is not None:
        with open(args.input_qrels, 'r') as r:
            for line in r:
                line = line.strip().split()
                if line[0] not in last_qds:
                    last_qds[line[0]] = {}
                last_qds[line[0]][line[2]] = 1

    f = open(args.output, 'w')
    qds = {}
    with open(args.input_trec, 'r') as r:
        for line in r:
            line = line.strip().split()
            if line[0] not in qds:
                qds[line[0]] = []
            if len(qds[line[0]]) >= args.topk:
                continue
            if line[0] in last_qds and line[2] in last_qds[line[0]]:
                continue
            else:
                qds[line[0]].append(line[2])
                f.write(' '.join(line) + '\n')
    f.close()

if __name__ == "__main__":
    main()
