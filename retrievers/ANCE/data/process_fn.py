import torch


def pad_ids(input_ids, attention_mask, token_type_ids, max_length, pad_token, mask_padding_with_zero, pad_token_segment_id, pad_on_left=False):
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1]
                          * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] *
                          padding_length) + token_type_ids
    else:
        input_ids += [pad_token] * padding_length
        attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
        token_type_ids += [pad_token_segment_id] * padding_length

    return input_ids, attention_mask, token_type_ids


def dual_process_fn(line, i, tokenizer, args):
    features = []
    cells = line.split("\t")
    if len(cells) == 2:
        # this is for training and validation
        # id, passage = line
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False

        text = cells[1].strip()
        input_id_a = tokenizer.encode(
            text, add_special_tokens=True, max_length=args.max_seq_length,)
        token_type_ids_a = [0] * len(input_id_a)
        attention_mask_a = [
            1 if mask_padding_with_zero else 0] * len(input_id_a)
        input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
            input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, tokenizer.pad_token_id, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
        features += [torch.tensor(input_id_a, dtype=torch.int), torch.tensor(
            attention_mask_a, dtype=torch.bool), torch.tensor(token_type_ids_a, dtype=torch.uint8)]
        qid = int(cells[0])
        features.append(qid)
    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 2.".format(str(len(cells))))
    return [features]


def triple_process_fn(line, i, tokenizer, args):
    features = []
    cells = line.split("\t")
    if len(cells) == 3:
        # this is for training and validation
        # query, positive_passage, negative_passage = line
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False

        for text in cells:
            input_id_a = tokenizer.encode(
                text.strip(), add_special_tokens=True, max_length=args.max_seq_length,)
            token_type_ids_a = [0] * len(input_id_a)
            attention_mask_a = [
                1 if mask_padding_with_zero else 0] * len(input_id_a)
            input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
                input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, tokenizer.pad_token_id, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
            features += [torch.tensor(input_id_a, dtype=torch.int),
                         torch.tensor(attention_mask_a, dtype=torch.bool)]
    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 3.".format(str(len(cells))))
    return [features]


def triple2dual_process_fn(line, i, tokenizer, args):
    ret = []
    cells = line.split("\t")
    if len(cells) == 3:
        # this is for training and validation
        # query, positive_passage, negative_passage = line
        # return 2 entries per line, 1 pos + 1 neg
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False
        pos_feats = []
        neg_feats = []

        for i, text in enumerate(cells):
            input_id_a = tokenizer.encode(
                text.strip(), add_special_tokens=True, max_length=args.max_seq_length,)
            token_type_ids_a = [0] * len(input_id_a)
            attention_mask_a = [
                1 if mask_padding_with_zero else 0] * len(input_id_a)
            input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
                input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, tokenizer.pad_token_id, mask_padding_with_zero, pad_token_segment_id, pad_on_left)
            if i == 0:
                pos_feats += [torch.tensor(input_id_a, dtype=torch.int),
                              torch.tensor(attention_mask_a, dtype=torch.bool)]
                neg_feats += [torch.tensor(input_id_a, dtype=torch.int),
                              torch.tensor(attention_mask_a, dtype=torch.bool)]
            elif i == 1:
                pos_feats += [torch.tensor(input_id_a, dtype=torch.int),
                              torch.tensor(attention_mask_a, dtype=torch.bool), 1]
            else:
                neg_feats += [torch.tensor(input_id_a, dtype=torch.int),
                              torch.tensor(attention_mask_a, dtype=torch.bool), 0]
        ret = [pos_feats, neg_feats]
    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 3.".format(str(len(cells))))
    return ret

