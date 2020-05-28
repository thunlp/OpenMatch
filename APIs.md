# API
APIs of OpenMatch.

## data
* OpenMatch.data.dataset(path, tokenizer, indexer, mode, delim)
    * mode: "train", "dev", "test".
    * delim: data split char, default ("\t").
* OpenMatch.data.Tokenizer(if\_swr, if\_lemma, if\_stem, max\_seq\_len)
    * if\_swr: using stopwords remove or not, default (False).
    * if\_lemma: using lemmatize or not, default (False).
    * if\_stem: using stem or not, default (False).
    * max\_seq\_len: max length of sequences.
    * Tokenizer.stopwords\_remove(tokens): remove stopwords in tokens.
    * Tokenizer.lemmatize(tokens): lemmatize tokens.
    * Tokenizer.stem(tokens): stem tokens.
    * Tokenizer.tokenize(text): tokenize raw text.
* OpenMatch.data.Indexer(path)
    * Indexer.index(tokens): generate indices and masks of tokens.

## metrics
* OpenMatch.metrics.Metric()
    * Metric.get\_metric(qrels, trec): evaluate trec with qrels.

## models
* OpenMatch.models.DSSM()
* OpenMatch.models.CDSSM()
* OpenMatch.models.K\_NRM()
* OpenMatch.models.Conv\_KNRM()
* OpenMatch.models.TK()
* OpenMatch.models.EDRM()
* OpenMatch.models.BERT()

## modules
* OpenMatch.modules.Embedder(vocab\_size, embed\_dim, embed\_matrix)
    * Embedder(indices): get embeddings of indices.
* OpenMatch.modules.Conv1DEncoder(embed\_dim, kernel\_dim, kernel\_sizes)
    * kernel\_dim: output dim of each kernel.
    * kernel\_sizes: sizes of kernels, default ([2, 3, 4, 5]).
    * Conv1DEncoder(embed, mask): encode embeddings to vector.
* OpenMatch.modules.KernelMatcher(embed\_dim, kernel\_num)
    * kernel\_num: number of kernels, default (21).
    * KernelMatcher(k\_embed, k\_mask, v\_embed, v\_mask): encode query, doc embeddings to ranking vector.
* OpenMatch.modules.MultiheadAttention(embed\_dim, head\_num, dropout)
    * MultiheadAttention(q\_embed, k\_embed, v\_embed): (N, L, D)
