import torch
import pandas as pd

train = pd.read_csv("data/train.csv", names=["Text", "Labels"])



class cfg:
    nlp_model = 'distilbert-base-uncased'
    inp = 768
    hidden = 256
    nlp_batch_size = 128
    nn_batch_size = 256
    num_workers = 4
    n_classes = len(train.Labels.unique())
    classes = {classes: value for value, classes in enumerate(train.Labels.unique())}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam
    criterion = torch.nn.CrossEntropyLoss()
    lr = 3e-4
    epochs = 50

del train