import sys
sys.path += ["../"]
from utils.msmarco_eval import quality_checks_qids, compute_metrics, load_reference
import torch.distributed as dist
import gzip
import faiss
import numpy as np
from data.process_fn import dual_process_fn
from tqdm import tqdm
import torch
import os
from utils.util import concat_key, is_first_worker, all_gather, StreamingDataset
from torch.utils.data import DataLoader


def embedding_inference(args, path, model, fn, bz, num_workers=2, is_query=True):
    f = open(path, encoding="utf-8")
    model = model.module if hasattr(model, "module") else model
    sds = StreamingDataset(f, fn)
    loader = DataLoader(sds, batch_size=bz, num_workers=0)
    emb_list, id_list = [], []
    model.eval()
    for i, batch in tqdm(enumerate(loader), desc="Eval", disable=args.local_rank not in [-1, 0]):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0].long(
            ), "attention_mask": batch[1].long()}
            idx = batch[3].long()
            if is_query:
                embs = model.query_emb(**inputs)
            else:
                embs = model.body_emb(**inputs)
            if len(embs.shape) == 3:
                B, C, E = embs.shape
                # [b1c1, b1c2, b1c3, b1c4, b2c1 ....]
                embs = embs.view(B*C, -1)
                idx = idx.repeat_interleave(C)

            assert embs.shape[0] == idx.shape[0]
            emb_list.append(embs.detach().cpu().numpy())
            id_list.append(idx.detach().cpu().numpy())
    f.close()
    emb_arr = np.concatenate(emb_list, axis=0)
    id_arr = np.concatenate(id_list, axis=0)

    return emb_arr, id_arr


def parse_top_dev(input_path, qid_col, pid_col):
    ret = {}
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            cells = line.strip().split("\t")
            qid = int(cells[qid_col])
            pid = int(cells[pid_col])
            if qid not in ret:
                ret[qid] = []
            ret[qid].append(pid)
    return ret


def search_knn(xq, xb, k, distance_type=faiss.METRIC_L2):
    """ wrapper around the faiss knn functions without index """
    nq, d = xq.shape
    nb, d2 = xb.shape
    assert d == d2

    I = np.empty((nq, k), dtype='int64')
    D = np.empty((nq, k), dtype='float32')

    if distance_type == faiss.METRIC_L2:
        heaps = faiss.float_maxheap_array_t()
        heaps.k = k
        heaps.nh = nq
        heaps.val = faiss.swig_ptr(D)
        heaps.ids = faiss.swig_ptr(I)
        faiss.knn_L2sqr(
            faiss.swig_ptr(xq), faiss.swig_ptr(xb),
            d, nq, nb, heaps
        )
    elif distance_type == faiss.METRIC_INNER_PRODUCT:
        heaps = faiss.float_minheap_array_t()
        heaps.k = k
        heaps.nh = nq
        heaps.val = faiss.swig_ptr(D)
        heaps.ids = faiss.swig_ptr(I)
        faiss.knn_inner_product(
            faiss.swig_ptr(xq), faiss.swig_ptr(xb),
            d, nq, nb, heaps
        )
    return D, I


def get_topk_restricted(q_emb, psg_emb_arr, pid_dict, psg_ids, pid_subset, top_k):
    subset_ix = np.array([pid_dict[x]
                          for x in pid_subset if x != -1 and x in pid_dict])
    if len(subset_ix) == 0:
        _D = np.ones((top_k,))*-128
        _I = (np.ones((top_k,))*-1).astype(int)
        return _D, _I
    else:
        sub_emb = psg_emb_arr[subset_ix]
        _D, _I = search_knn(q_emb, sub_emb, top_k,
                            distance_type=faiss.METRIC_INNER_PRODUCT)
        return _D.squeeze(), psg_ids[subset_ix[_I]].squeeze()  # (top_k,)


def passage_dist_eval(args, model, tokenizer):
    base_path = args.data_dir
    passage_path = os.path.join(base_path, "collection.tsv")
    queries_path = os.path.join(base_path, "queries.dev.small.tsv")

    def fn(line, i):
        return dual_process_fn(line, i, tokenizer, args)

    top1000_path = os.path.join(base_path, "top1000.dev.tsv")
    top1k_qid_pid = parse_top_dev(top1000_path, qid_col=0, pid_col=1)

    mrr_ref_path = os.path.join(base_path, "qrels.dev.small.tsv")
    ref_dict = load_reference(mrr_ref_path)

    reranking_mrr, full_ranking_mrr = combined_dist_eval(
        args, model, queries_path, passage_path, fn, fn, top1k_qid_pid, ref_dict)
    return reranking_mrr, full_ranking_mrr


def combined_dist_eval(args, model, queries_path, passage_path, query_fn, psg_fn, topk_dev_qid_pid, ref_dict):
    # get query/psg embeddings here
    eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    query_embs, query_ids = embedding_inference(
        args, queries_path, model, query_fn, eval_batch_size, 1, True)
    query_pkl = {"emb": query_embs, "id": query_ids}
    all_query_list = all_gather(query_pkl)
    query_embs = concat_key(all_query_list, "emb")
    query_ids = concat_key(all_query_list, "id")
    print(query_embs.shape, query_ids.shape)
    psg_embs, psg_ids = embedding_inference(
        args, passage_path, model, psg_fn, eval_batch_size, 2, False)
    print(psg_embs.shape)

    top_k = 100
    D, I = search_knn(query_embs, psg_embs, top_k,
                      distance_type=faiss.METRIC_INNER_PRODUCT)
    I = psg_ids[I]

    # compute reranking and full ranking mrr here
    # topk_dev_qid_pid is used for computing reranking mrr
    pid_dict = dict([(p, i) for i, p in enumerate(psg_ids)])
    arr_data = []
    d_data = []
    for i, qid in enumerate(query_ids):
        q_emb = query_embs[i:i+1]
        pid_subset = topk_dev_qid_pid[qid]
        ds, top_pids = get_topk_restricted(
            q_emb, psg_embs, pid_dict, psg_ids, pid_subset, 10)
        arr_data.append(top_pids)
        d_data.append(ds)
    _D = np.array(d_data)
    _I = np.array(arr_data)

    # reranking mrr
    reranking_mrr = compute_mrr(_D, _I, query_ids, ref_dict)
    D2 = D[:, :100]
    I2 = I[:, :100]
    # full mrr
    full_ranking_mrr = compute_mrr(D2, I2, query_ids, ref_dict)
    del psg_embs
    torch.cuda.empty_cache()
    #dist.barrier()
    return reranking_mrr, full_ranking_mrr


def compute_mrr(D, I, qids, ref_dict):
    knn_pkl = {"D": D, "I": I}
    all_knn_list = all_gather(knn_pkl)
    mrr = 0.0
    if is_first_worker():
        D_merged = concat_key(all_knn_list, "D", axis=1)
        I_merged = concat_key(all_knn_list, "I", axis=1)
        print(D_merged.shape, I_merged.shape)
        # we pad with negative pids and distance -128 - if they make it to the top we have a problem
        idx = np.argsort(D_merged, axis=1)[:, ::-1][:, :10]
        sorted_I = np.take_along_axis(I_merged, idx, axis=1)
        candidate_dict = {}
        for i, qid in enumerate(qids):
            seen_pids = set()
            if qid not in candidate_dict:
                candidate_dict[qid] = [0]*1000
            j = 0
            for pid in sorted_I[i]:
                if pid >= 0 and pid not in seen_pids:
                    candidate_dict[qid][j] = pid
                    j += 1
                    seen_pids.add(pid)

        allowed, message = quality_checks_qids(ref_dict, candidate_dict)
        if message != '':
            print(message)

        mrr_metrics = compute_metrics(ref_dict, candidate_dict)
        mrr = mrr_metrics["MRR @10"]
        print(mrr)
    return mrr
