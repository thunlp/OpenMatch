import pytrec_eval
import argparse
import csv
import json
import copy
import logging
import re
import unicodedata
from tqdm import tqdm
import numpy as np
from openmatch.utils import load_from_trec
from datasets import load_dataset

import regex

logger = logging.getLogger(__name__)


class Tokens(object):
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token
        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if 'pos' not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if 'lemma' not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if 'ner' not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [(s, e + 1)
                  for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        # Concatenate into strings
        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get('non_ent', 'O')
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag):
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
        if len(kwargs.get('annotators', {})) > 0:
            logger.warning('%s only tokenizes! Skipping annotators: %s' %
                           (type(self).__name__, kwargs.get('annotators')))
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append((
                token,
                text[start_ws: end_ws],
                span,
            ))
        return Tokens(data, self.annotators)


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def has_answers(text, answers, tokenizer, regex=False):
    text = _normalize(text)
    if regex:
        for ans in answers:
            ans = _normalize(ans)
            if regex_match(text, ans):
                return True
    else:
        text = tokenizer.tokenize(text).words(uncased=True)
        for ans in answers:
            ans = _normalize(ans)
            ans = tokenizer.tokenize(ans).words(uncased=True)
            for i in range(0, len(text) - len(ans) + 1):
                if ans == text[i: i + len(ans)]:
                    return True
    return False


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

    parser.add_argument("--qa", action="store_true")
    parser.add_argument("--collection", type=str, required=False)
    parser.add_argument("--answer", type=str, required=False)

    parser.add_argument("qrel")
    parser.add_argument("run")

    args = parser.parse_args()

    if args.qa:
        if args.collection is None or args.answer is None:
            raise ValueError("Must provide collection and answer files for QA eval")
        collection = {}
        # with open(args.collection, "r") as f:
        #     reader = csv.DictReader(f, delimiter="\t")
        #     for row in reader:
        #         id_ = row.pop("id")
        #         collection[id_] = row
        answer = {}
        answer_dataset = load_dataset("csv", data_files=[args.answer], column_names=["text","ans"], delimiter="\t")["train"]
        for item in answer_dataset:
            question, answers = item["text"], item["ans"]
            id_ = str(hash(question))
            answer[id_] = eval(answers)

        run = load_from_trec(args.run, as_list=True)

        tokenizer = SimpleTokenizer()
        accuracy = {1: [], 5: [], 20: [], 100: []}

        for qid, rank_list in run.items():
            answers = answer[qid]
            has_ans_idx = 100
            # for doc_rank, docid in enumerate(rank_list):
            #     text = collection[docid]["text"]
            #     if has_answers(text, answers, tokenizer):
            #         has_ans_idx = doc_rank
            #         break

            for k in accuracy:
                accuracy[k].append(int(has_ans_idx < k))

        for k in accuracy:
            print_line("Accuracy@{}".format(k), "all", np.mean(accuracy[k]))


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
