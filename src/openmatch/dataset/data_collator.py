# Adapted from Tevatron (https://github.com/texttron/tevatron)

from dataclasses import dataclass

from transformers import DataCollatorWithPadding, DefaultDataCollator


@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f["query"] for f in features]
        dd = [f["passages"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        return q_collated, d_collated


@dataclass
class PairCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        pos_pairs = [f["pos_pair"] for f in features]
        neg_pairs = [f["neg_pair"] for f in features]

        if isinstance(pos_pairs[0], list):
            pos_pairs = sum(pos_pairs, [])
        if isinstance(neg_pairs[0], list):
            neg_pairs = sum(neg_pairs, [])

        pos_pair_collated = self.tokenizer.pad(
            pos_pairs,
            padding='max_length',
            max_length=self.max_q_len + self.max_p_len + 2,
            return_tensors="pt",
        )
        neg_pair_collated = self.tokenizer.pad(
            neg_pairs,
            padding='max_length',
            max_length=self.max_q_len + self.max_p_len + 2,
            return_tensors="pt",
        )

        return pos_pair_collated, neg_pair_collated


@dataclass
class DRInferenceCollator(DefaultDataCollator):
    def __call__(self, features):
        text_ids = [x["text_id"] for x in features]
        collated_features = super().__call__(features)
        return text_ids, collated_features


@dataclass
class RRInferenceCollator(DefaultDataCollator):
    def __call__(self, features):
        query_ids = [x["query_id"] for x in features]
        doc_ids = [x["doc_id"] for x in features]
        collated_features = super().__call__(features)
        return query_ids, doc_ids, collated_features