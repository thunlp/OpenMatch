# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "2.11.0"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

import logging

# Configurations
from .configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
from .configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, CONFIG_MAPPING, AutoConfig
from .configuration_bart import BartConfig
from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
from .configuration_camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig
from .configuration_ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig
from .configuration_distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertConfig
from .configuration_electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig
from .configuration_encoder_decoder import EncoderDecoderConfig
from .configuration_flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig
from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config
from .configuration_longformer import LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LongformerConfig
from .configuration_marian import MarianConfig
from .configuration_mmbt import MMBTConfig
from .configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig
from .configuration_reformer import REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, ReformerConfig
from .configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig
from .configuration_t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config
from .configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
from .configuration_utils import PretrainedConfig
from .configuration_xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig
from .configuration_xlm_roberta import XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMRobertaConfig
from .configuration_xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig
from .data import (
    DataProcessor,
    InputExample,
    InputFeatures,
    SingleSentenceClassificationProcessor,
    SquadExample,
    SquadFeatures,
    SquadV1Processor,
    SquadV2Processor,
    glue_convert_examples_to_features,
    glue_output_modes,
    glue_processors,
    glue_tasks_num_labels,
    is_sklearn_available,
    squad_convert_examples_to_features,
    xnli_output_modes,
    xnli_processors,
    xnli_tasks_num_labels,
)

# Files and general utilities
from .file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_tf_available,
    is_torch_available,
)
from .hf_argparser import HfArgumentParser

# Model Cards
from .modelcard import ModelCard

# TF 2.0 <=> PyTorch conversion utilities
from .modeling_tf_pytorch_utils import (
    convert_tf_weight_name_to_pt_weight_name,
    load_pytorch_checkpoint_in_tf2_model,
    load_pytorch_model_in_tf2_model,
    load_pytorch_weights_in_tf2_model,
    load_tf2_checkpoint_in_pytorch_model,
    load_tf2_model_in_pytorch_model,
    load_tf2_weights_in_pytorch_model,
)

# Pipelines
from .pipelines import (
    CsvPipelineDataFormat,
    FeatureExtractionPipeline,
    FillMaskPipeline,
    JsonPipelineDataFormat,
    NerPipeline,
    PipedPipelineDataFormat,
    Pipeline,
    PipelineDataFormat,
    QuestionAnsweringPipeline,
    SummarizationPipeline,
    TextClassificationPipeline,
    TextGenerationPipeline,
    TokenClassificationPipeline,
    TranslationPipeline,
    pipeline,
)

# Tokenizers
from .tokenization_albert import AlbertTokenizer
from .tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from .tokenization_bart import BartTokenizer, MBartTokenizer
from .tokenization_bert import BasicTokenizer, BertTokenizer, BertTokenizerFast, WordpieceTokenizer
from .tokenization_bert_japanese import BertJapaneseTokenizer, CharacterTokenizer, MecabTokenizer
from .tokenization_camembert import CamembertTokenizer
from .tokenization_ctrl import CTRLTokenizer
from .tokenization_distilbert import DistilBertTokenizer, DistilBertTokenizerFast
from .tokenization_electra import ElectraTokenizer, ElectraTokenizerFast
from .tokenization_flaubert import FlaubertTokenizer
from .tokenization_gpt2 import GPT2Tokenizer, GPT2TokenizerFast
from .tokenization_longformer import LongformerTokenizer, LongformerTokenizerFast
from .tokenization_openai import OpenAIGPTTokenizer, OpenAIGPTTokenizerFast
from .tokenization_reformer import ReformerTokenizer
from .tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from .tokenization_t5 import T5Tokenizer
from .tokenization_transfo_xl import TransfoXLCorpus, TransfoXLTokenizer, TransfoXLTokenizerFast
from .tokenization_utils import PreTrainedTokenizer, TensorType
from .tokenization_xlm import XLMTokenizer
from .tokenization_xlm_roberta import XLMRobertaTokenizer
from .tokenization_xlnet import SPIECE_UNDERLINE, XLNetTokenizer
from .trainer_utils import EvalPrediction
from .training_args import TrainingArguments
from .training_args_tf import TFTrainingArguments


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


if is_sklearn_available():
    from .data import glue_compute_metrics, xnli_compute_metrics


# Modeling
if is_torch_available():
    from .modeling_utils import PreTrainedModel, prune_layer, Conv1D, top_k_top_p_filtering, apply_chunking_to_forward, ModuleUtilsMixin # add ModuleUtilsMixin (ss)
    from .modeling_auto import (
        AutoModel,
        AutoModelForPreTraining,
        AutoModelForSequenceClassification,
        AutoModelForQuestionAnswering,
        AutoModelWithLMHead,
        AutoModelForTokenClassification,
        AutoModelForMultipleChoice,
        MODEL_MAPPING,
        MODEL_FOR_PRETRAINING_MAPPING,
        MODEL_WITH_LM_HEAD_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    )

    from .modeling_bert import (
        BertPreTrainedModel,
        BertModel,
        BertForPreTraining,
        BertForMaskedLM,
        BertForNextSentencePrediction,
        BertForSequenceClassification,
        BertForMultipleChoice,
        BertForTokenClassification,
        BertForQuestionAnswering,
        load_tf_weights_in_bert,
        BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        BertLayer,
        BertSelfAttention,
    ) # add BertSelfAttention (ss)
    from .modeling_openai import (
        OpenAIGPTPreTrainedModel,
        OpenAIGPTModel,
        OpenAIGPTLMHeadModel,
        OpenAIGPTDoubleHeadsModel,
        load_tf_weights_in_openai_gpt,
        OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_transfo_xl import (
        TransfoXLPreTrainedModel,
        TransfoXLModel,
        TransfoXLLMHeadModel,
        AdaptiveEmbedding,
        load_tf_weights_in_transfo_xl,
        TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_gpt2 import (
        GPT2PreTrainedModel,
        GPT2Model,
        GPT2LMHeadModel,
        GPT2DoubleHeadsModel,
        load_tf_weights_in_gpt2,
        GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_ctrl import CTRLPreTrainedModel, CTRLModel, CTRLLMHeadModel, CTRL_PRETRAINED_MODEL_ARCHIVE_LIST
    from .modeling_xlnet import (
        XLNetPreTrainedModel,
        XLNetModel,
        XLNetLMHeadModel,
        XLNetForSequenceClassification,
        XLNetForTokenClassification,
        XLNetForMultipleChoice,
        XLNetForQuestionAnsweringSimple,
        XLNetForQuestionAnswering,
        load_tf_weights_in_xlnet,
        XLNET_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_xlm import (
        XLMPreTrainedModel,
        XLMModel,
        XLMWithLMHeadModel,
        XLMForSequenceClassification,
        XLMForTokenClassification,
        XLMForQuestionAnswering,
        XLMForQuestionAnsweringSimple,
        XLM_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_bart import (
        PretrainedBartModel,
        BartForSequenceClassification,
        BartModel,
        BartForConditionalGeneration,
        BART_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_marian import MarianMTModel
    from .tokenization_marian import MarianTokenizer
    from .modeling_roberta import (
        RobertaForMaskedLM,
        RobertaModel,
        RobertaForSequenceClassification,
        RobertaForMultipleChoice,
        RobertaForTokenClassification,
        RobertaForQuestionAnswering,
        ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_distilbert import (
        DistilBertPreTrainedModel,
        DistilBertForMaskedLM,
        DistilBertModel,
        DistilBertForSequenceClassification,
        DistilBertForQuestionAnswering,
        DistilBertForTokenClassification,
        DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_camembert import (
        CamembertForMaskedLM,
        CamembertModel,
        CamembertForSequenceClassification,
        CamembertForMultipleChoice,
        CamembertForTokenClassification,
        CamembertForQuestionAnswering,
        CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_encoder_decoder import EncoderDecoderModel
    from .modeling_t5 import (
        T5PreTrainedModel,
        T5Model,
        T5ForConditionalGeneration,
        load_tf_weights_in_t5,
        T5_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_albert import (
        AlbertPreTrainedModel,
        AlbertModel,
        AlbertForPreTraining,
        AlbertForMaskedLM,
        AlbertForSequenceClassification,
        AlbertForQuestionAnswering,
        AlbertForTokenClassification,
        load_tf_weights_in_albert,
        ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_xlm_roberta import (
        XLMRobertaForMaskedLM,
        XLMRobertaModel,
        XLMRobertaForMultipleChoice,
        XLMRobertaForSequenceClassification,
        XLMRobertaForTokenClassification,
        XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
    )
    from .modeling_mmbt import ModalEmbeddings, MMBTModel, MMBTForClassification

    from .modeling_flaubert import (
        FlaubertModel,
        FlaubertWithLMHeadModel,
        FlaubertForSequenceClassification,
        FlaubertForQuestionAnswering,
        FlaubertForQuestionAnsweringSimple,
        FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
    )

    from .modeling_electra import (
        ElectraForPreTraining,
        ElectraForMaskedLM,
        ElectraForTokenClassification,
        ElectraPreTrainedModel,
        ElectraForSequenceClassification,
        ElectraModel,
        load_tf_weights_in_electra,
        ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
    )

    from .modeling_reformer import (
        ReformerAttention,
        ReformerLayer,
        ReformerModel,
        ReformerModelWithLMHead,
        REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
    )

    from .modeling_longformer import (
        LongformerModel,
        LongformerForMaskedLM,
        LongformerForSequenceClassification,
        LongformerForMultipleChoice,
        LongformerForTokenClassification,
        LongformerForQuestionAnswering,
        LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
    )

    # Optimization
    from .optimization import (
        AdamW,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )

    # Trainer
    from .trainer import Trainer, set_seed, torch_distributed_zero_first, EvalPrediction
    from .data.data_collator import DefaultDataCollator, DataCollator, DataCollatorForLanguageModeling
    from .data.datasets import GlueDataset, TextDataset, LineByLineTextDataset, GlueDataTrainingArguments

    # Benchmarks
    from .benchmark import PyTorchBenchmark, PyTorchBenchmarkArguments

# TensorFlow
if is_tf_available():
    from .modeling_tf_utils import (
        shape_list,
        tf_top_k_top_p_filtering,
        TFPreTrainedModel,
        TFSequenceSummary,
        TFSharedEmbeddings,
    )
    from .modeling_tf_auto import (
        TF_MODEL_MAPPING,
        TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
        TF_MODEL_FOR_PRETRAINING_MAPPING,
        TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        TF_MODEL_WITH_LM_HEAD_MAPPING,
        TFAutoModel,
        TFAutoModelForMultipleChoice,
        TFAutoModelForPreTraining,
        TFAutoModelForQuestionAnswering,
        TFAutoModelForSequenceClassification,
        TFAutoModelForTokenClassification,
        TFAutoModelWithLMHead,
    )

    from .modeling_tf_albert import (
        TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFAlbertForMaskedLM,
        TFAlbertForMultipleChoice,
        TFAlbertForPreTraining,
        TFAlbertForQuestionAnswering,
        TFAlbertForSequenceClassification,
        TFAlbertForTokenClassification,
        TFAlbertMainLayer,
        TFAlbertModel,
        TFAlbertPreTrainedModel,
    )

    from .modeling_tf_bert import (
        TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFBertEmbeddings,
        TFBertForMaskedLM,
        TFBertForMultipleChoice,
        TFBertForNextSentencePrediction,
        TFBertForPreTraining,
        TFBertForQuestionAnswering,
        TFBertForSequenceClassification,
        TFBertForTokenClassification,
        TFBertMainLayer,
        TFBertModel,
        TFBertPreTrainedModel,
    )

    from .modeling_tf_camembert import (
        TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFCamembertForMaskedLM,
        TFCamembertModel,
        TFCamembertForMultipleChoice,
        TFCamembertForQuestionAnswering,
        TFCamembertForSequenceClassification,
        TFCamembertForTokenClassification,
    )

    from .modeling_tf_ctrl import (
        TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFCTRLLMHeadModel,
        TFCTRLModel,
        TFCTRLPreTrainedModel,
    )

    from .modeling_tf_distilbert import (
        TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFDistilBertForMaskedLM,
        TFDistilBertForMultipleChoice,
        TFDistilBertForQuestionAnswering,
        TFDistilBertForSequenceClassification,
        TFDistilBertForTokenClassification,
        TFDistilBertMainLayer,
        TFDistilBertModel,
        TFDistilBertPreTrainedModel,
    )

    from .modeling_tf_electra import (
        TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFElectraForMaskedLM,
        TFElectraForPreTraining,
        TFElectraForQuestionAnswering,
        TFElectraForTokenClassification,
        TFElectraModel,
        TFElectraPreTrainedModel,
    )

    from .modeling_tf_flaubert import (
        TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFFlaubertForMultipleChoice,
        TFFlaubertForQuestionAnsweringSimple,
        TFFlaubertForSequenceClassification,
        TFFlaubertForTokenClassification,
        TFFlaubertWithLMHeadModel,
        TFFlaubertModel,
    )

    from .modeling_tf_gpt2 import (
        TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFGPT2DoubleHeadsModel,
        TFGPT2LMHeadModel,
        TFGPT2MainLayer,
        TFGPT2Model,
        TFGPT2PreTrainedModel,
    )

    from .modeling_tf_openai import (
        TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFOpenAIGPTDoubleHeadsModel,
        TFOpenAIGPTLMHeadModel,
        TFOpenAIGPTMainLayer,
        TFOpenAIGPTModel,
        TFOpenAIGPTPreTrainedModel,
    )

    from .modeling_tf_roberta import (
        TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFRobertaForMaskedLM,
        TFRobertaForMultipleChoice,
        TFRobertaForQuestionAnswering,
        TFRobertaForSequenceClassification,
        TFRobertaForTokenClassification,
        TFRobertaMainLayer,
        TFRobertaModel,
        TFRobertaPreTrainedModel,
    )

    from .modeling_tf_t5 import (
        TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFT5ForConditionalGeneration,
        TFT5Model,
        TFT5PreTrainedModel,
    )

    from .modeling_tf_transfo_xl import (
        TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFAdaptiveEmbedding,
        TFTransfoXLLMHeadModel,
        TFTransfoXLMainLayer,
        TFTransfoXLModel,
        TFTransfoXLPreTrainedModel,
    )

    from .modeling_tf_xlm import (
        TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFXLMForMultipleChoice,
        TFXLMForQuestionAnsweringSimple,
        TFXLMForSequenceClassification,
        TFXLMForTokenClassification,
        TFXLMWithLMHeadModel,
        TFXLMMainLayer,
        TFXLMModel,
        TFXLMPreTrainedModel,
    )

    from .modeling_tf_xlm_roberta import (
        TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFXLMRobertaForMaskedLM,
        TFXLMRobertaForMultipleChoice,
        TFXLMRobertaForQuestionAnswering,
        TFXLMRobertaForSequenceClassification,
        TFXLMRobertaForTokenClassification,
        TFXLMRobertaModel,
    )

    from .modeling_tf_xlnet import (
        TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFXLNetForMultipleChoice,
        TFXLNetForQuestionAnsweringSimple,
        TFXLNetForSequenceClassification,
        TFXLNetForTokenClassification,
        TFXLNetLMHeadModel,
        TFXLNetMainLayer,
        TFXLNetModel,
        TFXLNetPreTrainedModel,
    )

    # Optimization
    from .optimization_tf import (
        AdamWeightDecay,
        create_optimizer,
        GradientAccumulator,
        WarmUp,
    )

    # Trainer
    from .trainer_tf import TFTrainer


if not is_tf_available() and not is_torch_available():
    logger.warning(
        "Neither PyTorch nor TensorFlow >= 2.0 have been found."
        "Models won't be available and only tokenizers, configuration"
        "and file/data utilities can be used."
    )
