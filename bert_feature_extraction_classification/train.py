import torch
from cfg import cfg


def validate(model, valid_loader):
    model.to(cfg.device)
    model.eval()

    with torch.no_grad():
        for batch in valid_loader:
            features, labels = tuple(t.to(cfg.device) for t in batch)
            outputs = model(features)
            loss = cfg.criterion(outputs, labels)
            accuracy = (outputs.argmax(dim=1) ==
                        labels.argmax(dim=1)).float().mean()
    print(f"Loss = {loss}  \t | Accuracy = {accuracy}")


def train(model, train_loader, valid_loader):
    model.to(cfg.device)
    model.train()

    optimizer = cfg.optimizer(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        net_loss, net_accuracy = 0, 0
        for batch in train_loader:
            features, labels = tuple(t.to(cfg.device) for t in batch)
            outputs = model(features)
            loss = cfg.criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            net_loss += loss
            net_accuracy += (outputs.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
        print(f"Loss = {net_loss/len(train_loader)}  \t | Accuracy = {net_accuracy/len(train_loader)}")

        validate(model, valid_loader)
    return model















