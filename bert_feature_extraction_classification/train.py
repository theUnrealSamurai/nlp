import torch
import wandb
from cfg import cfg


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, cfg, wandb=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = cfg.optimizer(model.parameters(), lr=cfg.lr)
        self.criterion = cfg.criterion
        self.cfg = cfg
        self.wandb = wandb

    def train_one_epoch(self, epoch):
        self.model.train()
        self.model.to(self.cfg.device)
        total_loss, total_accuracy = 0, 0
        for batch in self.train_loader:
            features, labels = tuple(t.to(self.cfg.device) for t in batch)
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss
            accuracy = (outputs.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
            total_accuracy += accuracy
            self.logger(epoch=epoch, loss=loss, accuracy=accuracy)
        self.logger(epoch=epoch, loss=total_loss/len(self.train_loader), accuracy=total_accuracy/len(self.train_loader), end='\n')
        return total_loss/len(self.train_loader), total_accuracy/len(self.train_loader)

    def validate(self, loader):
        self.model.eval()
        total_loss, total_accuracy = 0, 0
        with torch.no_grad():
            for batch in loader:
                features, labels = tuple(t.to(cfg.device) for t in batch)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss
                accuracy = (outputs.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
                total_accuracy += accuracy
            print(f"Validation Loss: {total_loss/len(loader)} | Validation Accuracy: {total_accuracy/len(loader)}")
            return total_loss/len(loader), total_accuracy/len(loader)

    def train(self):
        for epoch  in range(self.cfg.epochs):
            training_loss, training_accuracy = self.train_one_epoch(epoch)
            valid_loss, valid_accuracy = self.validate(self.val_loader)
            if self.wandb:
                wandb.log({"Epoch": epoch, 
                "TraininigLoss": training_loss,
                "TrainingAccuracy": training_accuracy,
                "ValidationLoss": valid_loss, 
                "ValidationAccuracy": valid_accuracy})
        return self.model 

    def test(self):
        self.model.eval()
        test_loss, test_accuracy = self.validate(self.test_loader)
        print(f"Test Loss: {test_loss} | Test Accuracy: {test_accuracy}")

    @staticmethod
    def logger(epoch, loss, accuracy, end='\r'):
        print(f'Epoch: {epoch} | Training Loss: {loss:.4f} | Training Accuracy: {accuracy:.4f}', end=end)