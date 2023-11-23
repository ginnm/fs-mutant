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


class SVMModel:
    def __init__(self, kernel):
        self.model = SVR(kernel=kernel)

    def train(self, X, Y):
        self.model.fit(X, Y)
        return self

    def pred(self, X):
        return self.model.predict(X)


class SVMTrainTest:
    def __init__(
        self,
        train_file,
        test_file,
        ensemble_models,
        score_col="score",
        kernel="rbf",
    ):
        self.train_file = train_file
        self.test_file = test_file
        self.ensemble_models = ensemble_models
        self.score_col = score_col
        self.kernel = kernel

    def train_test(self, model_name, save_dir):
        # 读取文件
        train_df = pd.read_csv(self.train_file)
        test_df = pd.read_csv(self.test_file)
        train_df = pd.merge(train_df[['mutant',]], test_df, on="mutant").reset_index()
        
        train_X = np.array([train_df[col] for col in self.ensemble_models]).T
        train_Y = train_df[self.score_col].values
        test_X = np.array([test_df[col] for col in self.ensemble_models]).T
        test_Y = test_df[self.score_col].values
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        x_scaler.fit(test_X)
        y_scaler.fit(test_Y.reshape(-1, 1))

        train_X = x_scaler.transform(train_X)
        train_Y = y_scaler.transform(train_Y.reshape(-1, 1)).reshape(-1)
        test_X = x_scaler.transform(test_X)
        test_Y = y_scaler.transform(test_Y.reshape(-1, 1)).reshape(-1)

        model = SVMModel(kernel=self.kernel).train(train_X, train_Y)
        pred = model.pred(X=test_X)

        test_df[model_name] = pred
        test_df = test_df[['mutant', model_name]]
        test_df.to_csv(os.path.join(save_dir, f"{model_name}.csv"), index=False)


def get_args():
    psr = ArgumentParser()
    psr.add_argument(
        "--train",
        type=str,
    )
    psr.add_argument(
        "--pred",
        type=str,
    )
    psr.add_argument(
        "--kernel",
        type=str,
        default='rbf'
    )
    psr.add_argument("--col", type=str, nargs="+")
    psr.add_argument("--save_dir", type=str)
    args = psr.parse_args()
    return args



def main():
    args = get_args()
    
    svm = SVMTrainTest(
        train_file=args.train,
        test_file=args.pred,
        ensemble_models=args.col,
        score_col="score",
        kernel=args.kernel,
    )
    
    train_file = Path(args.train)
    split_name = train_file.parent.parent.name
    train_size = train_file.parent.name.split("_")[0]
    split_iter = train_file.parent.name.split("_")[1]
    model_name = f"svm_{train_size}_{split_name}_{split_iter}"
    svm.train_test(model_name=model_name, save_dir=args.save_dir)
    return svm

if __name__ == "__main__":
    main()
