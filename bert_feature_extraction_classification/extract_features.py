import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel 
from tqdm import tqdm


class ClassificationDataset(Dataset):
    def __init__(self, df, cfg):
        self.df = df.values
        self.cfg = cfg

    def __len__(self):
        return self.df.__len__()

    def __getitem__(self, idx):
        text = self.df[idx][0]
        label = self.df[idx][1]
#         label = F.one_hot(torch.tensor([cfg.classes[label]]), num_classes=cfg.n_classes)
        # converting the labels to one hot array. faster than the above line.
        label = torch.tensor(np.eye(self.cfg.n_classes)[self.cfg.classes[label]])
        return {"text": text, "label": label}



def extract_features(df, cfg):
    dset = ClassificationDataset(df, cfg)
    dataloader = DataLoader(dset, batch_size=cfg.nlp_batch_size, shuffle=True, num_workers=cfg.num_workers)
    tokenizer = AutoTokenizer.from_pretrained(cfg.nlp_model)
    model = AutoModel.from_pretrained(cfg.nlp_model)

    features, labels = [], []
    model.to(cfg.device)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            tokens = tokenizer(
                batch['text'], padding=True, return_tensors='pt')
            tokens = tokens.to(cfg.device)
            outputs = model(**tokens)['last_hidden_state'][:, 0]

            features.append(outputs.detach().cpu().numpy())
            labels.append(batch['label'].detach().cpu().numpy())

    features = torch.tensor(np.concatenate(features))
    labels = torch.tensor(np.concatenate(labels), dtype=torch.float16)
    
    return {"features": features, "labels": labels}
