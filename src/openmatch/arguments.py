# Adapted from Tevatron (https://github.com/texttron/tevatron)

from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )
    feature: str = field(
        default="last_hidden_state",
        metadata={"help": "What feature to be extracted from the HF PLM"}
    )
    pooling: str = field(
        default="first",
        metadata={"help": "How to pool the features from the HF PLM"}
    )

    # out projection
    add_linear_head: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )

    encoder_only: bool = field(
        default=False,
        metadata={"help": "Whether to only use the encoder of T5"}
    )
    pos_token: Optional[str] = field(
        default=None,
        metadata={"help": "Token that indicates a relevant document"}
    )
    neg_token: Optional[str] = field(
        default=None,
        metadata={"help": "Token that indicates a irrelevant document"}
    )

    normalize: bool = field(
        default=False,
        metadata={"help": "Whether to normalize the embeddings"}
    )


@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: str = field(
        default=None, metadata={"help": "Path to single train file"}
    )
    eval_path: str = field(
        default=None, metadata={"help": "Path to eval file"}
    )
    query_path: str = field(
        default=None, metadata={"help": "Path to query file"}
    )
    corpus_path: str = field(
        default=None, metadata={"help": "Path to corpus file"}
    )
    data_dir: str = field(
        default=None, metadata={"help": "Path to data directory"}
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the single data file"}
    )
    processed_data_path: str = field(
        default=None, metadata={"help": "Path to processed data directory"}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    passage_field_separator: str = field(default=' ')
    dataset_proc_num: int = field(
        default=12, metadata={"help": "number of proc used in dataset preprocess"}
    )
    train_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"})

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )


    query_template: str = field(
        default="<text>",
        metadata={"help": "template for query"}
    )
    query_column_names: str = field(
        default="id,text",
        metadata={"help": "column names for the tsv data format"}
    )
    doc_template: str = field(
        default="Title: <title> Text: <text>",
        metadata={"help": "template for doc"}
    )
    doc_column_names: str = field(
        default="id,title,text",
        metadata={"help": "column names for the tsv data format"}
    )


@dataclass
class DRTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)


@dataclass
class RRTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    margin: float = field(default=1.0)
    loss_fn: str = field(
        default="bce",
        metadata={"help": "loss function to use"}
    )


@dataclass
class InferenceArguments(TrainingArguments):
    use_gpu: bool = field(default=False, metadata={"help": "Use GPU for Faiss retrieval"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    trec_save_path: str = field(default=None, metadata={"help": "where to save the trec file"})

    trec_run_path: str = field(default=None, metadata={"help": "previous stage TrecRun file"})
    id_key_name: str = field(default="id", metadata={"help": "key name for id"})

    reranking_depth: int = field(default=None, metadata={"help": "re-ranking depth"})