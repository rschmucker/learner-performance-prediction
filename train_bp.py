import pickle
import argparse
import pandas as pd
from random import shuffle
from sklearn.metrics import accuracy_score

import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

from utils import *


def accuracy_topk_scores(labels, preds, thresholds=[5, 100]):
    sorted = torch.argsort(preds, axis=-1)
    topk_acc = {f'top{k}_acc': (labels.unsqueeze(-1) == sorted[:, -k:]).sum().item() / len(labels)
                for k in thresholds}
    return topk_acc


@torch.no_grad()
def get_data(df, dkt, load_KS, train_split=0.8, KS_path='data/squirrel_ai/KS'):
    """Extract sequences from dataframe.

    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        dkt (torch Module): trained DKT model
        load_KS (bool): if True, load pickle file for knowledge states
        train_split (float): proportion of data to use for training
        KS_path (str): path of pickle file for knowledge states
    """
    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]

    if load_KS:
        with open(KS_path, 'rb') as f:
            KS = pickle.load(f)

    else:
        skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                     for _, u_df in df.groupby("user_id")]
        labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
                  for _, u_df in df.groupby("user_id")]

        item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i + 1))[:-1] for i in item_ids]
        skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s + 1))[:-1] for s in skill_ids]
        label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]

        data = list(zip(item_inputs, skill_inputs, label_inputs, labels))
        KS = []
        batch_size = 100

        for k in range(0, len(data), batch_size):
            batch = data[k:k + batch_size]
            seq_lists = list(zip(*batch))
            inputs = [pad_sequence(seqs, batch_first=True, padding_value=0).cuda()
                      for seqs in seq_lists[:-1]]
            labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)
            KS.append(dkt.get_KS(*inputs).cpu()[labels >= 0])

        KS = torch.cat(KS, dim=0)

        with open(KS_path, 'wb') as f:
            pickle.dump(KS, f, protocol=4)

    item_ids = torch.cat(item_ids, dim=0).flatten()

    # Train-test split across users
    train_size = int(train_split * KS.shape[0])
    X_train, X_val = KS[:train_size], KS[train_size:]
    y_train, y_val = item_ids[:train_size], item_ids[train_size:]
    return X_train, X_val, y_train, y_val


def train(X_train, X_val, y_train, y_val, policy, optimizer, logger, saver, num_epochs, batch_size):
    criterion = nn.CrossEntropyLoss()
    metrics = Metrics()
    train_idxs = np.arange(X_train.shape[0])
    val_idxs = np.arange(X_val.shape[0])
    step = 0

    for epoch in range(num_epochs):
        shuffle(train_idxs)
        shuffle(val_idxs)

        # Training
        for k in range(0, len(train_idxs), batch_size):
            X, y = X_train[train_idxs[k:k + batch_size]], y_train[train_idxs[k:k + batch_size]]
            y_hat = policy(X.cuda())
            train_loss = criterion(y_hat, y.cuda())
            train_acc = accuracy_score(y, y_hat.max(1)[1].cpu())
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
        for k in range(0, len(val_idxs), batch_size):
            X, y = X_val[val_idxs[k:k + batch_size]], y_val[val_idxs[k:k + batch_size]]
            with torch.no_grad():
                y_hat = policy(X.cuda())
                val_loss = criterion(y_hat, y.cuda())
                val_acc = accuracy_score(y, y_hat.max(1)[1].cpu())
                val_topk_acc = accuracy_topk_scores(y.cuda(), y_hat)
            metrics.store({'loss/val': val_loss.item()})
            metrics.store({'acc/val': val_acc})
            for k, v in val_topk_acc.items():
                metrics.store({f'{k}/val': v})

        # Save model
        average_metrics = metrics.average()
        logger.log_scalars(average_metrics, step)
        stop = saver.save(average_metrics['loss/val'], policy)
        if stop:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train behavior policy on Squirrel AI data set.')
    parser.add_argument('--logdir', type=str, default='runs/bp')
    parser.add_argument('--savedir', type=str, default='save/bp')
    parser.add_argument('--load_KS', action='store_true',
                        help='If True, load pickle file for knowledge states.')
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--KS_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_epochs', type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv('data/squirrel_ai/preprocessed_data.csv', sep="\t")
    dkt = torch.load('save/dkt2/squirrel_ai')

    X_train, X_val, y_train, y_val = get_data(df, dkt, args.load_KS)

    num_items = int(df["item_id"].max() + 1)

    policy = nn.Linear(args.KS_size, num_items).cuda()
    optimizer = Adam(policy.parameters(), lr=args.lr)

    logger = Logger(os.path.join(args.logdir, 'bp'))
    saver = Saver(args.savedir, 'bp')
    train(X_train, X_val, y_train, y_val, policy, optimizer, logger, saver, args.num_epochs,
          args.batch_size)
    logger.close()

