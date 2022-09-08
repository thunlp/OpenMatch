import pytrec_eval
import argparse


def eval_mrr(qrel, run, cutoff=None):
    """
    Compute MRR@cutoff manually.
    """
    mrr = 0.0
    num_ranked_q = 0
    results = {}
    for qid in qrel:
        if qid not in run:
            continue
        num_ranked_q += 1
        docid_and_score = [(docid, score) for docid, score in run[qid].items()]
        docid_and_score.sort(key=lambda x: x[1], reverse=True)
        for i, (docid, _) in enumerate(docid_and_score):
            rr = 0.0
            if cutoff is None or i < cutoff:
                if docid in qrel[qid] and qrel[qid][docid] > 0:
                    rr = 1.0 / (i + 1)
                    break
        results[qid] = rr
        mrr += rr
    mrr /= num_ranked_q
    results["all"] = mrr
    return results


def print_line(measure, scope, value):
    print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query_eval_wanted", action="store_true")
    parser.add_argument("-m", "--measure", type=str, default=None)
    parser.add_argument("qrel")
    parser.add_argument("run")
    args = parser.parse_args()

    with open(args.qrel, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    with open(args.run, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)


    if args.measure is not None and "mrr" in args.measure:

        if "mrr_cut" in args.measure:
            mrr_result = eval_mrr(qrel, run, cutoff=int(args.measure.split(".")[-1]))
        else:
            mrr_result = eval_mrr(qrel, run)
        if not args.query_eval_wanted:
            print("MRR: ", mrr_result["all"])
        else:
            for qid, mrr in mrr_result.items():
                print_line("MRR", qid, mrr)
            print("MRR: ", mrr_result["all"])

    else:
        if args.measure is None:
            evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_eval.supported_measures)
        else:
            evaluator = pytrec_eval.RelevanceEvaluator(qrel, {args.measure})
        results = evaluator.evaluate(run)

        for query_id, query_measures in sorted(results.items()):
            for measure, value in sorted(query_measures.items()):
                if args.query_eval_wanted:
                    print_line(measure, query_id, value)

        for measure in sorted(query_measures.keys()):
            print_line(
                measure,
                'all',
                pytrec_eval.compute_aggregated_measure(
                    measure,
                    [query_measures[measure] for query_measures in results.values()]))
