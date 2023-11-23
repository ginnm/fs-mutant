import torch
import torch.nn as nn
from argparse import ArgumentParser
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import numpy as np
from pathlib import Path
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import minmax_scale, normalize, StandardScaler
import pandas as pd
import os
from tqdm import tqdm
from utils import read_seq, full_sequence
from sklearn.linear_model import Ridge

AA_dict = "MRHKDESTNQCUGPAVIFYWLOXZBJ-"
AA2INT = {aa: i for i, aa in enumerate(AA_dict)}
INT2AA = {i: aa for i, aa in enumerate(AA_dict)}


def load_data(train_file, test_file, ):
    train_df = pd.read_csv(train_file)
    train_df = train_df[['mutant', ]]
    test_df = pd.read_csv(test_file)
    train_df = pd.merge(train_df, test_df, on="mutant").reset_index()
    return train_df, test_df

def seqs2feat(seqs, seq_zero_shot_scores=None):
    feat = np.zeros((len(seqs), len(seqs[0]) ,len(AA_dict))) # [B, L, A]
    for b, seq in enumerate(seqs):
        for  i, aa in enumerate(seq):
            feat[b, i, AA2INT[aa]] = 1
    feat = np.sum(feat, axis=1) # [B, A]
    if seq_zero_shot_scores is not None:
        feat = np.concatenate([feat, seq_zero_shot_scores.reshape(-1, 1)], axis=1) # [B, A+n, ]
    return feat

def train_agumented_model(
    fasta,
    train_file,
    test_file,
    add_cols=None,
    save_name=None,
    save_dir=None,
    svr=False
):
    wild_seq = read_seq(fasta)
    train_df, test_df = load_data(train_file, test_file)
    test_seqs = [full_sequence(wild_seq, m) for m in test_df['mutant']]
    train_seqs = [full_sequence(wild_seq, m) for m in train_df['mutant']]
    
    if add_cols:
        train_X = seqs2feat(train_seqs, seq_zero_shot_scores=train_df[add_cols].values.squeeze())
        test_X = seqs2feat(test_seqs, seq_zero_shot_scores=test_df[add_cols].values.squeeze())
    else:
        train_X = seqs2feat(train_seqs)
        test_X = seqs2feat(test_seqs)
    train_Y = train_df['score'].values
    test_Y = test_df['score'].values
    
    if svr:
        model = SVR(kernel="rbf")
    else:
        model = Ridge()
    model.fit(train_X, train_Y)
    pred = model.predict(test_X)
    train_file = Path(train_file)
    split_name = train_file.parent.parent.name
    train_size = train_file.parent.name.split("_")[0]
    split_iter = train_file.parent.name.split("_")[1]
    model_name = f"{save_name}_{train_size}_{split_name}_{split_iter}"
    test_df[model_name] = pred.squeeze()
    test_df = test_df[['mutant', model_name]]
    test_df.to_csv(os.path.join(save_dir, f"{model_name}.csv"), index=False)
    
def get_args():
    psr = ArgumentParser()
    psr.add_argument(
        "--fasta",
        type=str,
    )
    psr.add_argument(
        "--train",
        type=str,
    )
    psr.add_argument(
        "--test",
        type=str,
    )
    psr.add_argument(
        "--svr",
        action="store_true",
        default=False
    )
    psr.add_argument("--col", type=str, nargs="+")
    psr.add_argument("--save_dir", type=str)
    psr.add_argument("--model_name", type=str)
    args = psr.parse_args()
    return args


def main():
    args = get_args()

    train_agumented_model(
        fasta=args.fasta,
        train_file=args.train,
        test_file=args.test,
        add_cols=args.col,
        save_name=args.model_name,
        save_dir=args.save_dir,
        svr=args.svr
    )

if __name__ == "__main__":
    main()
