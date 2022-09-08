import os
import sys

import pytrec_eval
from openmatch.arguments import DataArguments
from openmatch.arguments import InferenceArguments as EncodingArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import BEIRDataset
from openmatch.modeling import DRModelForInference
from openmatch.retriever import Retriever
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, EncodingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, encoding_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, encoding_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        encoding_args: EncodingArguments

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    model = DRModelForInference.build(
        model_name_or_path=model_args.model_name_or_path,
        model_args=model_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    beir_dataset = BEIRDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        cache_dir=model_args.cache_dir
    )
    # it = iter(beir_dataset.query_dataset)
    # print(next(it))
    # print(next(it))
    # print(next(it))
    # print(next(it))
    # print(next(it))

    # exit(0)

    retriever = Retriever.build_all(model, beir_dataset.corpus_dataset, encoding_args)
    run = retriever.query_embedding_inference(beir_dataset.query_dataset)

    if encoding_args.local_process_index == 0:

        evaluator = pytrec_eval.RelevanceEvaluator(
        beir_dataset.qrel, {'ndcg_cut.10'})
        eval_results = evaluator.evaluate(run)

        def print_line(measure, scope, value):
            print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))

        for query_id, query_measures in sorted(eval_results.items()):
            for measure, value in sorted(query_measures.items()):
                pass

        # Scope hack: use query_measures of last item in previous loop to
        # figure out all unique measure names.
        #
        # TODO(cvangysel): add member to RelevanceEvaluator
        #                  with a list of measure names.
        for measure in sorted(query_measures.keys()):
            print_line(
                measure,
                'all',
                pytrec_eval.compute_aggregated_measure(
                    measure,
                    [query_measures[measure]
                    for query_measures in eval_results.values()]))


if __name__ == '__main__':
    main()
