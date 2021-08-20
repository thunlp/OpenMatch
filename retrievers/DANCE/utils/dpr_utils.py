import collections
import sys
sys.path += ['../']
import glob
import logging
import os
from typing import List, Tuple, Dict
import faiss
import pickle
import numpy as np
import unicodedata
import torch
import torch.distributed as dist
from torch import nn
from torch.serialization import default_restore_location
import regex
from transformers import AdamW
from utils.lamb import Lamb


logger = logging.getLogger()

CheckpointState = collections.namedtuple("CheckpointState",
                                         ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset', 'epoch',
                                          'encoder_params'])

def get_encoder_checkpoint_params_names():
    return ['do_lower_case', 'pretrained_model_cfg', 'encoder_model_type',
            'pretrained_file',
            'projection_dim', 'sequence_length']

def get_encoder_params_state(args):
    """
     Selects the param values to be saved in a checkpoint, so that a trained model faile can be used for downstream
     tasks without the need to specify these parameter again
    :return: Dict of params to memorize in a checkpoint
    """
    params_to_save = get_encoder_checkpoint_params_names()

    r = {}
    for param in params_to_save:
        r[param] = getattr(args, param)
    return r

def set_encoder_params_from_state(state, args):
    if not state:
        return
    params_to_save = get_encoder_checkpoint_params_names()

    override_params = [(param, state[param]) for param in params_to_save if param in state and state[param]]
    for param, value in override_params:
        if hasattr(args, param):
            logger.warning('Overriding args parameter value from checkpoint state. Param = %s, value = %s', param,
                           value)
        setattr(args, param, value)
    return args

def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, 'module') else model


def get_model_file(args, file_prefix) -> str:
    out_cp_files = glob.glob(os.path.join(args.output_dir, file_prefix + '*')) if args.output_dir else []
    logger.info('Checkpoint files %s', out_cp_files)
    model_file = None

    if args.model_file and os.path.exists(args.model_file):
        model_file = args.model_file
    elif len(out_cp_files) > 0:
        model_file = max(out_cp_files, key=os.path.getctime)
    return model_file


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)

def get_optimizer(args, model: nn.Module, weight_decay: float = 0.0, ) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.optimizer == "adamW":
        return AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    elif args.optimizer == "lamb":
        return Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        raise Exception("optimizer {0} not recognized! Can only be lamb or adamW".format(args.optimizer))


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.
    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.
    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
    """
    SIZE_STORAGE_BYTES = 4  # int32 to encode the payload size

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if enc_size + SIZE_STORAGE_BYTES > max_size:
        raise ValueError(
            'encoded data exceeds max_size, this can be fixed by increasing buffer size: {}'.format(enc_size))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    buffer_size = max_size * world_size

    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()

    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    assert enc_size < 256 ** SIZE_STORAGE_BYTES, 'Encoded object size should be less than {} bytes'.format(
        256 ** SIZE_STORAGE_BYTES)

    size_bytes = enc_size.to_bytes(SIZE_STORAGE_BYTES, byteorder='big')

    cpu_buffer[0:SIZE_STORAGE_BYTES] = torch.ByteTensor(list(size_bytes))
    cpu_buffer[SIZE_STORAGE_BYTES: enc_size + SIZE_STORAGE_BYTES] = torch.ByteTensor(list(enc))

    start = rank * max_size
    size = enc_size + SIZE_STORAGE_BYTES
    buffer[start: start + size].copy_(cpu_buffer[:size])

    if group is None:
        group = dist.group.WORLD
    dist.all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size: (i + 1) * max_size]
            size = int.from_bytes(out_buffer[0:SIZE_STORAGE_BYTES], byteorder='big')
            if size > 0:
                result.append(pickle.loads(bytes(out_buffer[SIZE_STORAGE_BYTES: size + SIZE_STORAGE_BYTES].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data.'
        )



class DenseHNSWFlatIndexer(object):
    """
     Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
    """

    def __init__(self, vector_sz: int, buffer_size: int = 50000, store_n: int = 512
                 , ef_search: int = 128, ef_construction: int = 200):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

        # IndexHNSWFlat supports L2 similarity only
        # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
        index = faiss.IndexHNSWFlat(vector_sz + 1, store_n)
        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = ef_construction
        self.index = index
        self.phi = 0

    def index_data(self, data: List[Tuple[object, np.array]]):
        n = len(data)

        # max norm is required before putting all vectors in the index to convert inner product similarity to L2
        if self.phi > 0:
            raise RuntimeError('DPR HNSWF index needs to index all data at once,'
                               'results will be unpredictable otherwise.')
        phi = 0
        for i, item in enumerate(data):
            id, doc_vector = item
            norms = (doc_vector ** 2).sum()
            phi = max(phi, norms)
        logger.info('HNSWF DotProduct -> L2 space phi={}'.format(phi))
        self.phi = 0

        # indexing in batches is beneficial for many faiss index types
        for i in range(0, n, self.buffer_size):
            db_ids = [t[0] for t in data[i:i + self.buffer_size]]
            vectors = [np.reshape(t[1], (1, -1)) for t in data[i:i + self.buffer_size]]

            norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
            aux_dims = [np.sqrt(phi - norm) for norm in norms]
            hnsw_vectors = [np.hstack((doc_vector, aux_dims[i].reshape(-1, 1))) for i, doc_vector in
                            enumerate(vectors)]
            hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)

            self._update_id_mapping(db_ids)
            self.index.add(hnsw_vectors)
            logger.info('data indexed %d', len(self.index_id_to_db_id))

        indexed_cnt = len(self.index_id_to_db_id)
        logger.info('Total data indexed %d', indexed_cnt)

    def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:

        aux_dim = np.zeros(len(query_vectors), dtype='float32')
        query_nhsw_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        logger.info('query_hnsw_vectors %s', query_nhsw_vectors.shape)
        scores, indexes = self.index.search(query_nhsw_vectors, top_docs)
        # convert to external ids
        db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs] for query_top_idxs in indexes]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

    def _update_id_mapping(self, db_ids: List):
        self.index_id_to_db_id.extend(db_ids)



def check_answer(passages, answers, doc_ids, tokenizer):
    """Search through all the top docs to see if they have any of the answers."""
    hits = []
    for i, doc_id in enumerate(doc_ids):
        text = passages[doc_id][0]
        hits.append(has_answer(answers, text, tokenizer))
    return hits


def has_answer(answers, text, tokenizer):
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """

    if text is None:
        logger.warning("no doc in db")
        return False

    text = _normalize(text)

    # Answer is a list of possible strings
    text = tokenizer.tokenize(text).words(uncased=True)

    for single_answer in answers:
        single_answer = _normalize(single_answer)
        single_answer = tokenizer.tokenize(single_answer)
        single_answer = single_answer.words(uncased=True)

        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i: i + len(single_answer)]:
                return True
    return False


class SimpleTokenizer:
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


def _normalize(text):
    return unicodedata.normalize('NFD', text)


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

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]
