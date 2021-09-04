# Contrastive Supervision Synthesis

Here provides the guiding code for running contrastive supervision synthesis technique, short for ContrastQG. A detailed introduction to the technology can be found in the paper [**Few-Shot Text Ranking with Meta Adapted Synthetic Weak Supervision**](https://arxiv.org/pdf/2012.14862.pdf).


### Source-domain NLG training.

We train two query generators (QG & ContrastQG) with the MS MARCO dataset using `train_nlg.sh`:

```
bash train_nlg.sh
```

Optional arguments:

```
--generator_mode            choices=['qg', 'contrastqg']
--pretrain_generator_type   choices=['t5-small', 't5-base']
--train_file                The path to the source-domain nlg training dataset
--save_dir                  The path to save the checkpoints data
```

### Target-domain NLG inferences

The whole nlg inference pipline contains five steps:

-   1/ Data preprocess
-   2/ Seed query generation
-   3/ BM25 subset retrieval
-   4/ Contrastive doc pairs sampling
-   5/ Contrastive query generation

1/ Data preprocess. convert target-domain documents into the nlg format
using `prepro_dataset.sh` in the folder `preprocess`:

    bash prepro_dataset.sh

Optional arguments:

```
--dataset_name The name of the target dataset
--input_path The path to the target dataset
--output_path The path to save the preprocess data
```

2/ Seed query generation. utilize the trained QG model to generate seed
queries for each target documents using `qg_inference.sh` in the folder
`run_shell`:

```
bash qg_inference.sh
```

Optional arguments:

```
--generator_mode choices='qg'
--pretrain_generator_type choices=['t5-small', 't5-base']
--target_dataset_name The name of the target dataset
--generator_load_dir The path to the pretrained QG checkpoints
```


3/ BM25 subset retrieval. utilize BM25 to retrieve document subset
according to the seed queries using the following shell commands in the
folder `bm25_retriever`:

```
bash build_index.sh
bash retrieve.sh
```

Optional arguments:

```
--dataset_name          The name of the target dataset
--data_path             The path to the target dataset
```

4/ Contrastive doc pairs sampling. pairwise sample contrastive doc pairs
from the BM25 retrieved subset using `sample_contrast_pairs.sh` in the
folder `preprocess`:

```
bash sample_contrast_pairs.sh
```

Optional arguments:

```
--dataset_name choices=['clueweb09', 'robust04', 'trec-covid']
--input_path The path to the target dataset
--generator_folder The path to the sampled data

5/ Contrastive query generation. utilize the trained ContrastQG model to
generate new queries based on contrastive document pairs using
`cqg_inference.sh` in the folder `run_shell`:

```
bash cqg_inference.sh
```

Optional arguments:

```
--generator_mode            choices='contrastqg'
--pretrain_generator_type   choices=['t5-small', 't5-base']
--target_dataset_name       The name of the target dataset
--generator_load_dir        The path to the pretrained ContrastQG checkpoints

```
