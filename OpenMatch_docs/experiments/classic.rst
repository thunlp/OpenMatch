Classic Features
================

We extract several classic IR features, and train learning-to-rank
models, such as RankSVM, Coor-Ascent, on ClueWeb09-B, Robust04 and
TREC-COVID datasets with 5 fold cross-validation. All the results can be
found in our `paper <https://arxiv.org/abs/2012.14862>`__ of ACL 2021.

The features consists of Boolean AND; Boolean OR; Coordinate match;
Cosine similarity of bag-of-words vectors; TF-IDF; BM25; language models
with no smoothing, Dirichlet smoothing, JM smoothing, and two-way
smoothing. More details are available at
`classic\_extractor <../OpenMatch/extractors/classic_extractor.py>`__.
