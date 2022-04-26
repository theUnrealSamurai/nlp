import torch
import pandas as pd

train = pd.read_csv("data/train.csv", names=["Text", "Labels"])

class features_config:
    model_name = 'distilbert-base-uncased'
    batch_size = 64
    num_workers = 4
    n_classes = len(train.Labels.unique())
    classes = {classes: value for value,
               classes in enumerate(train.Labels.unique())}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class cfg:
    inp = 768
    hidden = 256
    batch_size = 32
    num_workers = 4
    n_classes = len(train.Labels.unique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam
    criterion = torch.nn.CrossEntropyLoss()
    lr = 3e-4
    epochs = 50

del train