import logging
import os
import sys

from openmatch.arguments import DataArguments
from openmatch.arguments import InferenceArguments
from openmatch.arguments import ModelArguments
from openmatch.dataset import InferenceDataset
from openmatch.modeling import RRModel
from openmatch.retriever import Reranker
from openmatch.utils import save_as_trec, load_from_trec
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, InferenceArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, inference_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, inference_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        inference_args: InferenceArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if inference_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        inference_args.local_rank,
        inference_args.device,
        inference_args.n_gpu,
        bool(inference_args.local_rank != -1),
        inference_args.fp16,
    )
    logger.info("Encoding parameters %s", inference_args)
    logger.info("MODEL parameters %s", model_args)

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

    model = RRModel.build(
        model_args=model_args,
        tokenizer=tokenizer,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    query_dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        data_args=data_args,
        final=False,
        is_query=True,
        stream=False,
        cache_dir=model_args.cache_dir
    )

    corpus_dataset = InferenceDataset.load(
        tokenizer=tokenizer,
        data_args=data_args,
        final=False,
        is_query=False,
        stream=False,
        cache_dir=model_args.cache_dir
    )

    run = load_from_trec(inference_args.trec_run_path, max_len_per_q=inference_args.reranking_depth)

    reranker = Reranker(model, tokenizer, corpus_dataset, inference_args)
    result = reranker.rerank(query_dataset, run)

    if inference_args.local_process_index == 0:
        save_as_trec(result, inference_args.trec_save_path)


if __name__ == '__main__':
    main()
