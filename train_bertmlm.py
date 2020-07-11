import argparse

import torch
import torch.nn as nn

from transformers import AutoTokenizer, BertForMaskedLM, get_linear_schedule_with_warmup
import OpenMatch as om

def train(args, model, m_optim, m_scheduler, train_loader, device):
    for epoch in range(args.epoch):
        avg_loss = 0.0
        for step, train_batch in enumerate(train_loader):
            outputs = model(train_batch['input_ids'].to(device), train_batch['input_mask'].to(device), train_batch['segment_ids'].to(device), masked_lm_labels=train_batch['input_ids'].to(device))
            batch_loss = outputs[0]
            if torch.cuda.device_count() > 1:
                batch_loss = batch_loss.mean()
            avg_loss += batch_loss.item()
            batch_loss.backward()
            m_optim.step()
            m_scheduler.step()
            m_optim.zero_grad()

            if (step+1) % args.eval_every == 0:
                print(step+1, avg_loss/args.eval_every)
                model.save_pretrained(om.utils.check_dir(args.save + '_' + str(step+1)))
                avg_loss = 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action=om.utils.DictOrStr, default='./data/docs_toy.jsonl')
    parser.add_argument('-max_input', type=int, default=1280000)
    parser.add_argument('-save', type=str, default='./checkpoints/chkpt')
    parser.add_argument('-vocab', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-pretrain', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('-max_seq_len', type=int, default=256)
    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-n_warmup_steps', type=int, default=1000)
    parser.add_argument('-eval_every', type=int, default=1000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.vocab)
    print('reading training data...')
    train_set = om.data.datasets.BertMLMDataset(
        dataset=args.train,
        tokenizer=tokenizer,
        seq_max_len=args.max_seq_len,
        max_input=args.max_input,
    )

    train_loader = om.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8
    )

    model = BertForMaskedLM.from_pretrained(args.pretrain)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    m_scheduler = get_linear_schedule_with_warmup(m_optim, num_warmup_steps=args.n_warmup_steps, num_training_steps=len(train_set)*args.epoch//args.batch_size)

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train(args, model, m_optim, train_loader, device)

if __name__ == "__main__":
    main()
