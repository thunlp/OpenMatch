from os.path import join
import sys
sys.path += ['../']
import argparse
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from model.models import MSMarcoConfigDict, ALL_MODELS
import csv
from utils.util import multi_file_process, numbered_byte_file_generator, EmbeddingCache
import pickle


def normalize_question(question: str) -> str:
    if question[-1] == '?':
        question = question[:-1]
    return question


def write_qas_query(args, qas_file, out_query_file):
    print("Writing qas query files " + str(out_query_file))
    print("print",args.answer_dir,qas_file)
    qas_path = os.path.join(
        args.answer_dir,
        qas_file,
    )
    out_query_path = os.path.join(
        args.out_data_dir,
        out_query_file ,
    )

    configObj = MSMarcoConfigDict[args.model_type]
    tokenizer = configObj.tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True,
        cache_dir=None,
    )

    qid = 0
    with open(qas_path, "r", encoding="utf-8") as f, open(out_query_path, "wb") as out_query:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            question = normalize_question(row[0])
            out_query.write(QueryPreprocessingFn(args, qid, question, tokenizer))
            qid += 1

    meta = {'type': 'int32', 'total_number': qid, 'embedding_size': args.max_seq_length}
    with open(out_query_path + "_meta", 'w') as f:
        json.dump(meta, f)


def write_query_rel(args, pid2offset, query_file, out_query_file, out_ann_file, out_train_file, passage_id_name="passage_id"):

    print("Writing query files " + str(out_query_file) + " and " + str(out_ann_file))

    query_path = os.path.join(
        args.question_dir,
        query_file,
    )

    with open(query_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        print('Aggregated data size: {}'.format(len(data)))

    data = [r for r in data if len(r['positive_ctxs']) > 0]
    print('Total cleaned data size: {}'.format(len(data)))
    data = [r for r in data if len(r['hard_negative_ctxs']) > 0]
    print('Total cleaned data size: {}'.format(len(data)))

    out_query_path = os.path.join(
        args.out_data_dir,
        out_query_file ,
    )

    out_ann_file = os.path.join(
        args.out_data_dir,
        out_ann_file ,
    )

    out_training_path = os.path.join(
        args.out_data_dir,
        out_train_file ,
    )

    qid = 0

    configObj = MSMarcoConfigDict[args.model_type]
    tokenizer = configObj.tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True,
        cache_dir=None,
    )

    with open(out_query_path, "wb") as out_query, \
            open(out_ann_file, "w", encoding='utf-8') as out_ann, \
            open(out_training_path, "w", encoding='utf-8') as out_training:
        for sample in data:
            positive_ctxs = sample['positive_ctxs']
            neg_ctxs = sample['hard_negative_ctxs']
            question = normalize_question(sample['question'])
            first_pos_pid = pid2offset[int(positive_ctxs[0][passage_id_name])]
            neg_pids = [str(pid2offset[int(neg_ctx[passage_id_name])]) for neg_ctx in neg_ctxs]
            out_ann.write("{}\t{}\t{}\n".format(qid, first_pos_pid, sample["answers"]))
            out_training.write("{}\t{}\t{}\n".format(qid, first_pos_pid, ','.join(neg_pids)))
            out_query.write(QueryPreprocessingFn(args, qid, question, tokenizer))
            qid += 1

    print("Total lines written: " + str(qid))
    meta = {'type': 'int32', 'total_number': qid, 'embedding_size': args.max_seq_length}
    with open(out_query_path + "_meta", 'w') as f:
        json.dump(meta, f)

    embedding_cache = EmbeddingCache(out_query_path)
    print("First line")
    with embedding_cache as emb:
        print(emb[0])


def write_mapping(args, id2offset, out_name):
    out_path = os.path.join(
        args.out_data_dir,
        out_name ,
    )
    with open(out_path, 'w') as f:
        for item in id2offset.items():
            f.write("{}\t{}\n".format(item[0], item[1]))


def load_mapping(data_dir, out_name):
    out_path = os.path.join(
        data_dir,
        out_name ,
    )
    pid2offset = {}
    offset2pid = {}
    with open(out_path, 'r') as f:
        for line in f.readlines():
            line_arr = line.split('\t')
            pid2offset[int(line_arr[0])] = int(line_arr[1])
            offset2pid[int(line_arr[1])] = int(line_arr[0])
    return pid2offset, offset2pid


def preprocess(args):

    pid2offset = {}
    in_passage_path = os.path.join(
        args.wiki_dir,
        "psgs_w100.tsv" ,
    )
    out_passage_path = os.path.join(
        args.out_data_dir,
        "passages" ,
    )

    if os.path.exists(out_passage_path):
        print("preprocessed data already exist, exit preprocessing")
        return
    else:
        out_line_count = 0

        print('start passage file split processing')
        multi_file_process(args, 32, in_passage_path, out_passage_path, PassagePreprocessingFn)

        print('start merging splits')
        with open(out_passage_path, 'wb') as f:
            for idx, record in enumerate(numbered_byte_file_generator(out_passage_path, 32, 8 + 4 + args.max_seq_length * 4)):
                p_id = int.from_bytes(record[:8], 'big')
                f.write(record[8:])
                pid2offset[p_id] = idx
                if idx < 3:
                    print(str(idx) + " " + str(p_id))
                out_line_count += 1

        print("Total lines written: " + str(out_line_count))
        meta = {'type': 'int32', 'total_number': out_line_count, 'embedding_size': args.max_seq_length}
        with open(out_passage_path + "_meta", 'w') as f:
            json.dump(meta, f)    
        write_mapping(args, pid2offset, "pid2offset")

    embedding_cache = EmbeddingCache(out_passage_path)
    print("First line")
    with embedding_cache as emb:
        print(emb[pid2offset[1]])

    if args.data_type == 0:
        write_query_rel(args, pid2offset, "nq-train.json", "train-query", "train-ann", "train-data")
    elif args.data_type == 1:
        write_query_rel(args, pid2offset, "trivia-train.json", "train-query", "train-ann", "train-data", "psg_id")
    else:
        # use both training dataset and merge them
        write_query_rel(args, pid2offset, "nq-train.json", "train-query-nq", "train-ann-nq", "train-data-nq")
        write_query_rel(args, pid2offset, "trivia-train.json", "train-query-trivia", "train-ann-trivia", "train-data-trivia", "psg_id")

        with open(args.out_data_dir + "train-query-nq", "rb") as nq_query, \
                open(args.out_data_dir + "train-query-trivia", "rb") as trivia_query, \
                open(args.out_data_dir + "train-query", "wb") as out_query:
            out_query.write(nq_query.read())
            out_query.write(trivia_query.read())

        with open(args.out_data_dir + "train-query-nq_meta", "r", encoding='utf-8') as nq_query, \
                open(args.out_data_dir + "train-query-trivia_meta", "r", encoding='utf-8') as trivia_query, \
                open(args.out_data_dir + "train-query_meta", "w", encoding='utf-8') as out_query:
            a = json.load(nq_query)
            b = json.load(trivia_query)
            meta = {'type': 'int32', 'total_number': a['total_number'] + b['total_number'], 'embedding_size': args.max_seq_length}
            json.dump(meta, out_query)

        embedding_cache = EmbeddingCache(args.out_data_dir + "train-query")
        print("First line after merge")
        with embedding_cache as emb:
            print(emb[58812])

        with open(args.out_data_dir + "train-ann-nq", "r", encoding='utf-8') as nq_ann, \
                open(args.out_data_dir + "train-ann-trivia", "r", encoding='utf-8') as trivia_ann, \
                open(args.out_data_dir + "train-ann", "w", encoding='utf-8') as out_ann:
            out_ann.writelines(nq_ann.readlines())
            out_ann.writelines(trivia_ann.readlines())


    write_query_rel(args, pid2offset, "nq-dev.json", "dev-query", "dev-ann", "dev-data")
    write_query_rel(args, pid2offset, "trivia-dev.json", "dev-query-trivia", "dev-ann-trivia", "dev-data-trivia", "psg_id")
    write_qas_query(args, "nq-test.csv", "test-query")
    write_qas_query(args, "trivia-test.csv", "trivia-test-query")

def PassagePreprocessingFn(args, line, tokenizer):
    line_arr = list(csv.reader([line], delimiter='\t'))[0]
    if line_arr[0] == 'id':
        return bytearray()

    p_id = int(line_arr[0])
    text = line_arr[1]
    title = line_arr[2]

    token_ids = tokenizer.encode(title, text_pair=text, add_special_tokens=True,
                                      max_length=args.max_seq_length,
                                      pad_to_max_length=False)

    seq_len = args.max_seq_length
    passage_len = len(token_ids)
    if len(token_ids) < seq_len:
        token_ids = token_ids + [tokenizer.pad_token_id] * (seq_len - len(token_ids))
    if len(token_ids) > seq_len:
        token_ids = token_ids[0:seq_len]
        token_ids[-1] = tokenizer.sep_token_id

    if p_id < 5:
        a = np.array(token_ids, np.int32)
        print("pid {}, passagelen {}, shape {}".format(p_id, passage_len, a.shape))

    return p_id.to_bytes(8, 'big') + passage_len.to_bytes(4, 'big') + np.array(token_ids, np.int32).tobytes()


def QueryPreprocessingFn(args, qid, text, tokenizer):
    token_ids = tokenizer.encode(text, add_special_tokens=True, max_length=args.max_seq_length,
                                       pad_to_max_length=False)

    seq_len = args.max_seq_length
    passage_len = len(token_ids)
    if len(token_ids) < seq_len:
        token_ids = token_ids + [tokenizer.pad_token_id] * (seq_len - len(token_ids))
    if len(token_ids) > seq_len:
        token_ids = token_ids[0:seq_len]
        token_ids[-1] = tokenizer.sep_token_id

    if qid < 5:
        a = np.array(token_ids, np.int32)
        print("qid {}, passagelen {}, shape {}".format(qid, passage_len, a.shape))

    return passage_len.to_bytes(4, 'big') + np.array(token_ids, np.int32).tobytes()


def GetProcessingFn(args, query=False):
    def fn(vals, i):
        passage_len, passage = vals
        max_len = args.max_seq_length
        
        pad_len = max(0, max_len - passage_len)
        token_type_ids = [0] * passage_len + [0] * pad_len
        attention_mask = passage != 0

        passage_collection = [(i, passage, attention_mask, token_type_ids)]
        
        query2id_tensor = torch.tensor([f[0] for f in passage_collection], dtype=torch.long)
        all_input_ids_a = torch.tensor([f[1] for f in passage_collection], dtype=torch.int)
        all_attention_mask_a = torch.tensor([f[2] for f in passage_collection], dtype=torch.bool)
        all_token_type_ids_a = torch.tensor([f[3] for f in passage_collection], dtype=torch.uint8)

        dataset = TensorDataset(all_input_ids_a, all_attention_mask_a, all_token_type_ids_a, query2id_tensor)

        return [ts for ts in dataset]
    
    return fn


def GetTrainingDataProcessingFn(args, query_cache, passage_cache, shuffle=True):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        all_input_ids_a = []
        all_attention_mask_a = []

        query_data = GetProcessingFn(args, query=True)(query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(args, query=False)(passage_cache[pos_pid], pos_pid)[0]

        if shuffle:
            random.shuffle(neg_pids)

        neg_data = GetProcessingFn(args, query=False)(passage_cache[neg_pids[0]], neg_pids[0])[0]
        yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2])
        yield (query_data[0], query_data[1], query_data[2], neg_data[0], neg_data[1], neg_data[2])

    return fn


def GetTripletTrainingDataProcessingFn(args, query_cache, passage_cache, shuffle=True):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        all_input_ids_a = []
        all_attention_mask_a = []

        query_data = GetProcessingFn(args, query=True)(query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(args, query=False)(passage_cache[pos_pid], pos_pid)[0]

        if shuffle:
            random.shuffle(neg_pids)

        neg_data = GetProcessingFn(args, query=False)(passage_cache[neg_pids[0]], neg_pids[0])[0]
        yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2], 
            neg_data[0], neg_data[1], neg_data[2])

    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_data_dir",
        default="/webdata-nfs/jialliu/dpr/ann/ann_multi_data_256/",
        type=str,
        help="The output data dir",
    )
    parser.add_argument(
        "--model_type",
        default="dpr",
        type=str,
        help="Model type selected in the list: " + ", ".join(MSMarcoConfigDict.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-uncased",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: " +
        ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--data_type",
        default=0,
        type=int,
        help="0 is nq, 1 is trivia, 2 is both",
    )
    parser.add_argument(
        "--question_dir",
        type=str,
        help="location of the raw QnA question data",
    )
    parser.add_argument(
        "--wiki_dir",
        type=str,
        help="location of the wiki corpus",
    )
    parser.add_argument(
        "--answer_dir",
        type=str,
        help="location of the QnA answers for evaluation",
    )
    args = parser.parse_args()
    if not os.path.exists(args.out_data_dir):
        os.makedirs(args.out_data_dir)
    preprocess(args)


if __name__ == '__main__':
    main()
