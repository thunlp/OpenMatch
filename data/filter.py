import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_qrels', type=str)
    parser.add_argument('-input_trec', type=str)
    parser.add_argument('-output', type=str)
    args = parser.parse_args()

    docs = {}
    with open(args.input_qrels, 'r') as r:
        for line in r:
            line = line.strip().split()
            docs[line[2]] = 1

    f = open(args.output, 'w')
    with open(args.input_trec, 'r') as r:
        for line in r:
            line = line.strip().split()
            if line[2] not in docs:
                f.write(' '.join(line) + '\n')
    f.close()

if __name__ == "__main__":
    main()
