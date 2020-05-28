from typing import List, Dict

import pytrec_eval

class Metric():
    def get_metric(self, qrels: str, trec: str) -> Dict[str, float]:
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
        return mes
