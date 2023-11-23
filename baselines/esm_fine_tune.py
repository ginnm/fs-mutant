import torch
import torch.nn as nn
from argparse import ArgumentParser
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import numpy as np
from pathlib import Path

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        if args.add_col:
            input_size = 1280 + len(args.add_col)
            self.add_fc = nn.Linear(len(args.add_col), len(args.add_col))
        else:
            input_size = 1280
            
        self.fc1 = nn.Linear(input_size, 1280)
        self.fc2 = nn.Linear(1280, 1)
        self.dropout = nn.Dropout(0.1)
        self.tanh = nn.Tanh()

    def forward(self, token_embeddings, labels=None):
        if self.args.add_col:
            add_values = self.add_fc(token_embeddings[:, 1280:])
            token_embeddings = torch.cat([token_embeddings[:, :1280], add_values], dim=1)
            x = self.fc1(token_embeddings)
            x = self.tanh(x)
            x = self.dropout(x)
            preds = self.fc2(x).squeeze()
        else:
            x = self.fc1(token_embeddings)
            x = self.tanh(x)
            x = self.dropout(x)
            preds = self.fc2(x).squeeze()
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(preds, labels.squeeze())
            return loss, preds
        else:
            return None, preds


def get_args():
    psr = ArgumentParser()
    psr.add_argument("--train", type=str, default="data")
    psr.add_argument("--pred", type=str, default="data")
    psr.add_argument("--model_name", type=str, default="data")
    psr.add_argument("--add_col", type=str, default=[], nargs="+")
    psr.add_argument("--train_embeddings", type=str, default="data")
    psr.add_argument("--pred_embeddings", type=str, default="data")
    psr.add_argument("--batch_size", type=int, default=32)
    psr.add_argument("--lr", type=float, default=1e-4)
    psr.add_argument("--epochs", type=int, default=100)
    psr.add_argument("--gpu", action="store_true", default=False)
    psr.add_argument("--save_dir", type=str)
    args = psr.parse_args()
    return args


def load_data(args):
    df = pd.read_csv(args.train)
    mutants = df['mutant'].values
    embeddings_dict = torch.load(args.train_embeddings)
    X = torch.stack([embeddings_dict[mutant] for mutant in mutants], dim=0)
    if args.add_col:
        pdf = pd.read_csv(args.pred).set_index('mutant')
        values = pdf.loc[mutants][args.add_col].values
        col = torch.from_numpy(values).to(torch.float)
        X = torch.cat([X, col], dim=1)
    
    Y = torch.from_numpy(df['score'].values).to(torch.float)
    dataset = TensorDataset(X, Y)
    for fold in range(5):
        if len(Y) <= 40:
            test_size = 10
        else:
            test_size = 0.2
        train_idx, valid_idx = train_test_split(range(len(Y)), test_size=test_size, random_state=fold)
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        valid_dataset = torch.utils.data.Subset(dataset, valid_idx)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        yield train_dataloader, valid_dataloader
        
def train(train_dataloader, valid_dataloader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.gpu else "cpu"
    
    best_state = {
        'epoch': 0,
        'spearmanr': -1,
        'model_state_dict': None,
        'optimizer_state_dict': None,
        'args' : args,
    }
    
    model = Model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            loss, preds = model(batch[0].to(device), batch[1].unsqueeze(1).to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds_list = []
            labels_list = []
            for batch in valid_dataloader:
                loss, preds = model(batch[0].to(device), batch[1].unsqueeze(1).to(device))
                preds_list.append(preds.cpu().detach().numpy())
                labels_list.append(batch[1].cpu().detach().numpy())
            spearmanr_score = spearmanr(np.concatenate(preds_list), np.concatenate(labels_list))[0]
            if spearmanr_score > best_state['spearmanr']:
                best_state['epoch'] = epoch
                best_state['spearmanr'] = spearmanr_score
                best_state['model_state_dict'] = model.state_dict()
                best_state['optimizer_state_dict'] = optimizer.state_dict()
        print(f"Epoch {epoch} spearmanr: {spearmanr_score}")
    return best_state

def ensemble_predict(args, best_states):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = []
    df = pd.read_csv(args.pred)
    mutants = df['mutant'].values
    embeddings_dict = torch.load(args.pred_embeddings)
    X = torch.stack([embeddings_dict[mutant] for mutant in mutants], dim=0)
    if args.add_col:
        col = torch.from_numpy(df[args.add_col].values).to(torch.float)
        X = torch.cat([X, col], dim=1)
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    for best_state in best_states:
        model = Model(args).to(device)
        model.load_state_dict(best_state['model_state_dict'])
        model.eval()
        with torch.no_grad():
            preds_list = []
            for batch in dataloader:
                _, preds = model(batch[0].to(device))
                preds_list.append(preds.cpu().detach().numpy())
            scores.append(np.concatenate(preds_list))
    scores = np.mean(scores, axis=0)
    
    train_file = Path(args.train)
    split_name = train_file.parent.parent.name
    train_size = train_file.parent.name.split("_")[0]
    split_iter = train_file.parent.name.split("_")[1]
    
    df[f"{args.model_name}_{train_size}_{split_name}_{split_iter}"] = scores
    df = df[['mutant', f"{args.model_name}_{train_size}_{split_name}_{split_iter}"]]
    df.to_csv(Path(args.save_dir) / f"{args.model_name}_{train_size}_{split_name}_{split_iter}.csv", index=False)

def main():
    args = get_args()
    best_states = []
    for train_dataloader, valid_dataloader in load_data(args):
        best_states.append(train(train_dataloader, valid_dataloader, args))
    ensemble_predict(args, best_states)
    
if __name__ == "__main__":
    main()