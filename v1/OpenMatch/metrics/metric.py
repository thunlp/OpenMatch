from typing import List, Dict

import pytrec_eval

class Metric():
    def get_metric(self, qrels: str, trec: str, metric: str = 'ndcg_cut_10') -> Dict[str, float]:
        with open(qrels, 'r') as f_qrel:
            qrel = pytrec_eval.parse_qrel(f_qrel)
        with open(trec, 'r') as f_run:
            run = pytrec_eval.parse_run(f_run)

        evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_eval.supported_measures)
        results = evaluator.evaluate(run)
        for query_id, query_measures in sorted(results.items()):
            pass
        mes = {}
        for measure in sorted(query_measures.keys()):
            mes[measure] = pytrec_eval.compute_aggregated_measure(measure, [query_measures[measure] for query_measures in results.values()])
        return mes[metric]

    def get_mrr(self, qrels: str, trec: str, metric: str = 'mrr_cut_10') -> float:
        k = int(metric.split('_')[-1])

        qrel = {}
        with open(qrels, 'r') as f_qrel:
            for line in f_qrel:
                qid, _, did, label = line.strip().split()
                if qid not in qrel:
                    qrel[qid] = {}
                qrel[qid][did] = int(label)

        run = {}
        with open(trec, 'r') as f_run:
            for line in f_run:
                qid, _, did, _, _, _ = line.strip().split()
                if qid not in run:
                    run[qid] = []
                run[qid].append(did)
        
        mrr = 0.0
        for qid in run:
            rr = 0.0
            for i, did in enumerate(run[qid][:k]):
                if qid in qrel and did in qrel[qid] and qrel[qid][did] > 0:
                    rr = 1 / (i+1)
                    break
            mrr += rr
        mrr /= len(run)
        return mrr
