from typing import List, Tuple

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class Tokenizer():
    def __init__(
        self,
        vocab: str = None,
        pretrained: str = None,
        if_swr: bool = True,
        if_stem: bool = True,
        sp_tok: str = '[PAD]'
    ) -> None:
        self._vocab = vocab
        self._pretrained = pretrained
        self._if_swr = if_swr
        self._if_stem = if_stem
        self._sp_tok = sp_tok

        if self._if_swr:
            self._stopwords = set(stopwords.words('english'))
        if self._if_stem:
            self._stemmer = PorterStemmer()
        
        self._id2token = None
        self._token2id = None
        self._embed_matrix = None
        if self._pretrained is not None:
            self.from_pretrained(self._pretrained)
        elif self._vocab is not None:
            self.from_vocab(self._vocab)
        else:
            raise ValueError('Tokenizer must be initialized with vocab.')

    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError('function tokenize not implemented')

    def process(self, text: str, max_len: int) -> Tuple[List[int], List[int]]:
        tokens = self.tokenize(text)
        if self._if_swr:
            tokens = self.stopwords_remove(tokens, max_len)
        if self._if_stem:
            tokens = self.stem(tokens)
        if len(tokens) < max_len:
            tokens = tokens + [self._sp_tok] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        ids = self.convert_tokens_to_ids(tokens)
        masks = [0 if tid == 0 else 1 for tid in ids]
        return ids, masks

    def token_process(self, tokens: List[str], max_len: int) -> Tuple[List[int], List[int]]:
        if len(tokens) < max_len:
            tokens = tokens + [self._sp_tok] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        ids = self.convert_tokens_to_ids(tokens)
        masks = [0 if tid == 0 else 1 for tid in ids]
        return ids, masks

    def batch_process(self, texts: List[str], max_len: int, max_num: int) -> Tuple[List[int], List[int]]:
        if len(texts) < max_num:
            texts = texts + [self._sp_tok] * (max_num - len(texts))
        else:
            texts = texts[:max_num]
        batch_ids, batch_masks = zip(*[self.process(text, max_len) for text in texts])
        return batch_ids, batch_masks

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self._token2id[token] if token in self._token2id else 0 for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self._id2token[tid] if tid in self._id2token else self._sp_tok for tid in ids]

    def from_vocab(self, vocab: str) -> None:
        tid = 0
        self._id2token = {tid: self._sp_tok}
        self._token2id = {self._sp_tok: tid}
        tid += 1
        with open(vocab, 'r') as reader:
            for line in reader:
                line = line.strip('\n')
                self._id2token[tid] = line
                self._token2id[line] = tid
                tid += 1

    def from_pretrained(self, pretrained: str) -> None:
        tid = 0
        self._id2token = {tid: self._sp_tok}
        self._token2id = {self._sp_tok: tid}
        self._embed_matrix = []
        tid += 1
        with open(self._pretrained, 'r') as reader:
            for line in reader:
                line = line.strip().split()
                self._id2token[tid] = line[0]
                self._token2id[line[0]] = tid
                self._embed_matrix.append([float(l) for l in line[1:]])
                tid += 1
        self._embed_matrix.insert(0, [0] * len(self._embed_matrix[0]))

    def get_vocab_size(self) -> int:
        return len(self._token2id) if self._token2id is not None else -1

    def get_embed_dim(self) -> int:
        return len(self._embed_matrix[0]) if self._embed_matrix is not None else -1

    def get_embed_matrix(self) -> List[List[float]]:
        return self._embed_matrix

    def stopwords_remove(self, tokens: List[str], max_len: int) -> List[str]:
        removed = []
        for token in tokens:
            if token not in self._stopwords:
                removed.append(token)
                if len(removed) >= max_len:
                    break
        return removed

    def stem(self, tokens: List[str]) -> List[str]:
        return [self._stemmer.stem(token) for token in tokens]
