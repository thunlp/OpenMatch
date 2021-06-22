import argparse
import json
import re
import math

import numpy as np
from scipy import spatial
from nltk.corpus import stopwords
from scipy.linalg import norm

class ClassicExtractor():
    # classical feature extractor
    def __init__(self, query_terms, doc_terms, df, total_df=None, avg_doc_len=None):
        """
        :param query_terms: query term -> tf
        :param doc_terms: doc term -> tf
        :param df: term -> df dict
        :param total_df: a int of total document frequency
        :param avg_doc_len: a float of avg document length
        """
        query_tf = [item[1] for item in query_terms.items()]
        query_df = []
        doc_tf = []
        for item in query_terms.items():
            if item[0] in df:
                query_df.append(df[item[0]])
            else:
                query_df.append(0)
            if item[0] in doc_terms:
                doc_tf.append(doc_terms[item[0]])
            else:
                doc_tf.append(0)
        
        self.query_tf = np.array(query_tf)
        self.query_df = np.array(query_df)
        self.doc_tf = np.array(doc_tf)

        self.doc_len = sum([item[1] for item in doc_terms.items()])
        if total_df is not None:
            self.total_df = total_df
        if avg_doc_len is not None:
            self.avg_doc_len = avg_doc_len

        self.k1 = 1.2
        self.b = 0.75
        self.dir_mu = 2500
        self.min_tf = 0.1
        self.jm_lambda = 0.4
        self.min_score = 1e-10
        return

    def get_feature(self):
        # l_sim_func = ['lm', 'lm_dir', 'lm_jm', 'lm_twoway',
        #               'bm25', 'coordinate', 'cosine', 'tf_idf',
        #               'bool_and', 'bool_or']
        features = {}
        features['lm'] = self.lm()
        features['lm_dir'] = self.lm_dir()
        features['lm_jm'] = self.lm_jm()
        features['lm_twoway'] = self.lm_twoway()
        features['bm25'] = self.bm25()
        features['coordinate'] = self.coordinate()
        features['cosine'] = self.cosine()
        features['tf_idf'] = self.tf_idf()
        features['bool_and'] = self.bool_and()
        features['bool_or'] = self.bool_or()
        return features

    def lm(self):
        if self.doc_len == 0:
            return np.log(self.min_score)
        v_tf = np.maximum(self.doc_tf, self.min_tf)
        v_tf /= self.doc_len
        v_tf = np.maximum(v_tf, self.min_score)
        score = np.log(v_tf).dot(self.query_tf)
        return score

    def lm_dir(self):
        if self.doc_len == 0:
            return np.log(self.min_score)
        v_q = self.query_tf / np.sum(self.query_tf)
        v_mid = (self.doc_tf + self.dir_mu * (self.query_df / self.total_df)) / (self.doc_len + self.dir_mu)
        v_mid = np.maximum(v_mid, self.min_score)
        score = np.log(v_mid).dot(v_q)
        return score

    def lm_jm(self):
        if self.doc_len == 0:
            return np.log(self.min_score)
        v_mid = self.doc_tf / self.doc_len * (1 - self.jm_lambda) + self.jm_lambda * self.query_df / self.total_df
        v_mid = np.maximum(v_mid, self.min_score)
        score = np.log(v_mid).dot(self.query_tf)
        return score

    def lm_twoway(self):
        if self.doc_len == 0:
            return np.log(self.min_score)
        v_mid = (self.doc_tf + self.dir_mu * (self.query_df / self.total_df)) / (self.doc_len + self.dir_mu)
        v_mid = v_mid * (1 - self.jm_lambda) + self.jm_lambda * self.query_df / self.total_df
        v_mid = np.maximum(v_mid, self.min_score)
        score = np.log(v_mid).dot(self.query_tf)
        return score

    def bm25(self):
        if self.doc_len == 0:
            return 0
        v_q = self.query_tf / float(np.sum(self.query_tf))
        v_tf_part = self.doc_tf * (self.k1 + 1) / (self.doc_tf + self.k1 * (1 - self.b + self.b * self.doc_len / self.avg_doc_len))
        v_mid = (self.total_df - self.query_df + 0.5) / (self.query_df + 0.5)
        v_mid = np.maximum(v_mid, 1.0)
        v_idf_q = np.log(v_mid)
        v_idf_q = np.maximum(v_idf_q, 0)
        score = v_mid.dot(v_tf_part * v_idf_q)
        score = max(score, 1.0)
        score = np.log(score)
        return score

    def cosine(self):
        if self.doc_len == 0:
            return 0
        if sum(self.doc_tf) == 0:
            return 0
        v_q = self.query_tf / float(np.sum(self.query_tf))
        v_d = self.doc_tf / float(self.doc_len)
        score = spatial.distance.cosine(v_q, v_d)
        if math.isnan(score):
            return 0
        return score

    def coordinate(self):
        return sum(self.doc_tf > 0)

    def bool_and(self):
        if self.coordinate() == len(self.query_tf):
            return 1
        return 0

    def bool_or(self):
        return min(1, self.coordinate())

    def tf_idf(self):
        if self.doc_len == 0:
            return 0
        normed_idf = np.log(1 + self.total_df / np.maximum(self.query_df, 1))
        normed_tf = self.doc_tf / self.doc_len
        return normed_idf.dot(normed_tf)
    
class Corpus():
    # corpus processor
    def __init__(self, docs):
        """
        :param docs: doc dict -> docs[docid] = doc
        """
        self.regex_drop_char = re.compile('[^a-z0-9\s]+')
        self.regex_multi_space = re.compile('\s+')
        self.stop_words = set(stopwords.words('english'))
        
        self.docs = docs

    def text2lm(self, text):
        tokens = self.regex_multi_space.sub(' ', self.regex_drop_char.sub(' ', text.lower())).strip().split()
        text_len = len(tokens)
        d = {}
        for token in tokens:
            if token not in d:
                d[token] = 0
            d[token] += 1
        return d, text_len

    def cnt_corpus(self):
        docs_terms = {}
        df = {}
        total_df = len(self.docs)
        total_doc_len = 0
        for doc in docs:
            doc_terms, doc_len = self.text2lm(self.docs[doc])
            docs_terms[doc] = doc_terms
            for item in doc_terms:
                if item not in df:
                    df[item] = 0
                df[item] += 1
            total_doc_len += doc_len
        avg_doc_len = total_doc_len / total_df
        return docs_terms, df, total_df, avg_doc_len
