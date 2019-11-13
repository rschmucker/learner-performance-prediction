import torch
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Write predictions to a dataframe.')
    parser.add_argument('--X_file', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--nn_path', type=str)
    parser.add_argument('--max_iter', type=int, default=1000)
    args = parser.parse_args()

    nn = torch.load(args.nn_path)

    # Load sparse dataset for logistic regression
    X = csr_matrix(load_npz(args.X_file))

    user_ids = X[:, 0].toarray().flatten()
    users_train = pd.read_csv(f'data/{args.dataset}/preprocessed_data_train.csv', sep="\t")["user_id"].unique()
    users_test = pd.read_csv(f'data/{args.dataset}/preprocessed_data_test.csv', sep="\t")["user_id"].unique()

    X_train = X[np.where(np.isin(user_ids, users_train))]
    X_test = X[np.where(np.isin(user_ids, users_test))]
    y_train = X_train[:, 3].toarray().flatten()
    y_test = X_test[:, 3].toarray().flatten()

    # Train logistic regression
    model = LogisticRegression(solver="lbfgs", max_iter=args.max_iter)
    model.fit(X_train[:, 5:], y_train)
    lr_preds = model.predict_proba(X_test[:, 5:])[:, 1]

    lr_data = np.concatenate((X_test[:, :5].toarray(), lr_preds.reshape(-1, 1)), axis=1)
    lr_df = pd.DataFrame(data=lr_data, columns=["user_id", "item_id", "timestamp", "correct", "skill_id", "LR"])

    full_data = None

    # Evaluate neural network
    for _, u_df in lr_df.groupby("user_id"):
        item_ids = torch.tensor(u_df["item_id"].values, dtype=torch.long)
        skill_ids = torch.tensor(u_df["skill_id"].values, dtype=torch.long)
        labels = torch.tensor(u_df["correct"].values, dtype=torch.long)

        item_inputs = torch.cat((torch.zeros(1, dtype=torch.long), item_ids + 1))[:-1]
        skill_inputs = torch.cat((torch.zeros(1, dtype=torch.long), skill_ids + 1))[:-1]
        label_inputs = torch.cat((torch.zeros(1, dtype=torch.long), labels + 1))[:-1]

        item_inputs = item_inputs.unsqueeze(0).cuda()
        skill_inputs = skill_inputs.unsqueeze(0).cuda()
        label_inputs = label_inputs.unsqueeze(0).cuda()
        item_ids = item_ids.unsqueeze(0).cuda()
        skill_ids = skill_ids.unsqueeze(0).cuda()

        with torch.no_grad():
            nn_preds, _ = nn(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            nn_preds = torch.sigmoid(nn_preds).cpu().squeeze(0)

        new_data = np.hstack((u_df.values, nn_preds.numpy().reshape(-1, 1)))
        full_data = new_data if full_data is None else np.vstack((full_data, new_data))

    full_df = pd.DataFrame(data=full_data,
                           columns=["user_id", "item_id", "timestamp", "correct", "skill_id", "LR", "NN"])

    lr_auc = roc_auc_score(full_df["correct"], full_df["LR"])
    nn_auc = roc_auc_score(full_df["correct"], full_df["NN"])
    print(f"{args.dataset}: lr_auc={lr_auc}, nn_auc={nn_auc}")

    full_df.to_csv(f'data/{args.dataset}/predictions_test.csv', sep="\t", index=False)
