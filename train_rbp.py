import argparse
import pandas as pd
from random import shuffle
from sklearn.metrics import accuracy_score

import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

from model_rbp import RBP
from utils import *


def get_data(df, train_split=0.8):
    """Extract sequences from dataframe.

    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        train_split (float): proportion of data to use for training
    """
    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]

    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i + 1))[:-1] for i in item_ids]
    skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s + 1))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]

    data = list(zip(item_inputs, skill_inputs, label_inputs, item_ids))
    shuffle(data)

    # Train-test split across users
    train_size = int(train_split * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    return train_data, val_data


def compute_acc(preds, labels):
    preds = preds[labels >= 0]
    labels = labels[labels >= 0]
    return accuracy_score(labels, preds)


def compute_loss(preds, labels, criterion):
    preds = preds[labels >= 0]
    labels = labels[labels >= 0]
    return criterion(preds, labels)


def compute_accuracy_topk_scores(labels, preds, thresholds=[5, 100]):
    preds = preds[labels >= 0]
    labels = labels[labels >= 0]
    sorted = torch.argsort(preds, axis=-1)
    topk_acc = {f'top{k}_acc': (labels.unsqueeze(-1) == sorted[:, -k:]).sum().item() / len(labels)
                for k in thresholds}
    return topk_acc


def prepare_batches(data, batch_size):
    """Prepare batches grouping padded sequences.

    Arguments:
        data (list of lists of torch Tensor): output by get_data
        batch_size (int): number of sequences per batch

    Output:
        batches (list of lists of torch Tensor)
    """
    shuffle(data)
    batches = []

    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        seq_lists = list(zip(*batch))
        inputs = [pad_sequence(seqs, batch_first=True, padding_value=0)
                  for seqs in seq_lists[:-1]]
        item_ids = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
        batches.append([*inputs, item_ids])

    return batches


def train(train_data, val_data, policy, optimizer, logger, saver, num_epochs, batch_size, bptt=50):
    criterion = nn.CrossEntropyLoss()
    metrics = Metrics()
    step = 0

    for epoch in range(num_epochs):
        train_batches = prepare_batches(train_data, batch_size)
        val_batches = prepare_batches(val_data, batch_size)

        # Training
        for item_inputs, skill_inputs, label_inputs, item_ids in train_batches:
            length = item_ids.size(1)
            preds = torch.empty(item_ids.size(0), length, policy.num_items)
            preds = preds.cuda()
            item_inputs = item_inputs.cuda()
            skill_inputs = skill_inputs.cuda()
            label_inputs = label_inputs.cuda()

            # Truncated backprop through time
            for i in range(0, length, bptt):
                item_inp = item_inputs[:, i:i + bptt]
                skill_inp = skill_inputs[:, i:i + bptt]
                label_inp = label_inputs[:, i:i + bptt]
                if i == 0:
                    pred, hidden = policy(item_inp, skill_inp, label_inp)
                else:
                    hidden = policy.repackage_hidden(hidden)
                    pred, hidden = policy(item_inp, skill_inp, label_inp, hidden)
                preds[:, i:i + bptt] = pred

            train_loss = compute_loss(preds, item_ids.cuda(), criterion)
            train_acc = compute_acc(torch.softmax(preds, -1).max(-1)[1].detach().cpu(), item_ids)

            policy.zero_grad()
            train_loss.backward()
            optimizer.step()
            step += 1
            metrics.store({'loss/train': train_loss.item()})
            metrics.store({'acc/train': train_acc})

            # Logging
            if step % 20 == 0:
                logger.log_scalars(metrics.average(), step)

        # Validation
        policy.eval()
        for item_inputs, skill_inputs, label_inputs, item_ids in val_batches:
            with torch.no_grad():
                item_inputs = item_inputs.cuda()
                skill_inputs = skill_inputs.cuda()
                label_inputs = label_inputs.cuda()
                preds, _ = policy(item_inputs, skill_inputs, label_inputs)
            val_loss = compute_loss(preds, item_ids.cuda(), criterion)
            val_acc = compute_acc(torch.softmax(preds, -1).max(-1)[1].cpu(), item_ids)
            val_topk_acc = compute_accuracy_topk_scores(item_ids.cuda(), preds)
            metrics.store({'loss/val': val_loss.item()})
            metrics.store({'acc/val': val_acc})
            for k, v in val_topk_acc.items():
                metrics.store({f'{k}/val': v})
        policy.train()

        # Save model
        average_metrics = metrics.average()
        logger.log_scalars(average_metrics, step)
        stop = saver.save(average_metrics['loss/val'], policy)
        if stop:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train recurrent behavior policy on Squirrel AI data set.')
    parser.add_argument('--logdir', type=str, default='runs/rbp')
    parser.add_argument('--savedir', type=str, default='save/rbp')
    parser.add_argument('--hid_size', type=int, default=200)
    parser.add_argument('--embed_size', type=int, default=200)
    parser.add_argument('--num_hid_layers', type=int, default=1)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_epochs', type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv('data/squirrel_ai/preprocessed_data.csv', sep="\t")

    train_data, val_data = get_data(df)

    num_items = int(df["item_id"].max() + 1)
    num_skills = int(df["skill_id"].max() + 1)

    policy = RBP(num_items, num_skills, args.hid_size, args.embed_size, args.num_hid_layers,
                 args.drop_prob).cuda()
    optimizer = Adam(policy.parameters(), lr=args.lr)

    # Reduce batch size until it fits on GPU
    while True:
        try:
            logger = Logger(os.path.join(args.logdir, f'rbp,batch_size={args.batch_size}'))
            saver = Saver(args.savedir, 'rbp')
            train(train_data, val_data, policy, optimizer, logger, saver, args.num_epochs, args.batch_size)
            break
        except RuntimeError:
            args.batch_size = args.batch_size // 2
            print(f'Batch does not fit on gpu, reducing size to {args.batch_size}')

    logger.close()
