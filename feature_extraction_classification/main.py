import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from train import train, validate  
from torch.utils.data import DataLoader, TensorDataset
from extract_features import extract_features
from cfg import features_config, cfg
from transformers import logging


def set_seed(seed):
    """Sets the random seed for the code reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class ClassificationModel(nn.Module):
    def __init__(self, inp, hidden, output):
        super(ClassificationModel, self).__init__()
        self.linear = torch.nn.Linear(inp, hidden)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden, output)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':

    set_seed(42)

    debug = False
    logging.set_verbosity_warning()
    logging.set_verbosity_error()
    
    train_df = pd.read_csv("data/train.csv", names=["Text", "Labels"])
    test_df = pd.read_csv("data/test.csv",  names=["Text", "Labels"])
    val_df = pd.read_csv("data/val.csv", names = ["Text", "Labels"])

    if debug:
        train_df = train_df[:250]    
        test_df = test_df[:250]
        val_df = val_df[:250]

    train_df = extract_features(train_df, features_config)
    val_df = extract_features(val_df, features_config)
    # test_df = extract_features(test_df, features_config)


    train_loader = DataLoader(TensorDataset(train_df['features'], train_df['labels']))
    val_loader = DataLoader(TensorDataset(val_df['features'], val_df['labels']))
    # test_loader = DataLoader(TensorDataset(test_df['features'], test_df['labels']))

    model = ClassificationModel(cfg.inp, cfg.hidden, cfg.n_classes)
    model = train(model, train_loader, val_loader)

