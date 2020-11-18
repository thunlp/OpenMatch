import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dev', type=str, default='../data/dev_toy.jsonl')
    parser.add_argument('-res', type=str, default='../results/cknrm_ca.trec')
    parser.add_argument('-k', type=int, default=2)
    args = parser.parse_args()

    score_dic = {}
    for i in range(args.k):
        with open('f' + str(i+1) + '.score', 'r') as r:
            for line in r:
                line = line.strip('\n').split('\t')
                score_dic[line[0] + '$' + line[1]] = line[2]

    if args.k == -1:
        with open('f' + str(args.k+1) + '.score', 'r') as r:
            for line in r:
                line = line.strip('\n').split('\t')
                score_dic[line[0] + '$' + line[1]] = line[2]

    outs = {}
    with open(args.dev, 'r') as r:
        qid = ''
        cnt = 0
        for line in r:
            line = json.loads(line)
            if line['query_id'] != qid:
                qid = line['query_id']
                cnt = 0
                outs[line['query_id']] = {}
            outs[line['query_id']][line['doc_id']] = float(score_dic[line['query_id']+'$'+str(cnt)])
            cnt += 1

    f = open(args.res, 'w')
    for qid in outs:
        ps = {}
        out_idx = sorted(outs[qid].items(), key=lambda x:x[1], reverse=True)
        for i, out in enumerate(out_idx):
            if out[0] not in ps:
                ps[out[0]] = 1
                f.write(' '.join([qid, 'Q0', out[0], str(len(ps)), str(out[1]), 'default']) + '\n')
    f.close()

if __name__ == "__main__":
    main()
