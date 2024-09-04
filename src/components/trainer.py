from src.entity.config_entity import TrainerConfig
from src.components.data_preprocessing import DataProcessing
from torch import nn
import numpy as np
import torch
import tqdm

class Trainer:

    def __init__(self, loaders: dict, device: str, net) -> None:
        self.config = TrainerConfig()
        self.trainLoader = loaders["train_data_loader"][0]
        self.testLoader = loaders["test_data_loader"][0]
        self.valLoader = loaders["val_data_loader"][0]
        self.device = device
        self.criterian = nn.CrossEntropyLoss()
        self.model = net.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.evaluation = self.config.Evaluation

    def train_model(self):

        print('Starting Training...')

        for epoch in range(self.config.EPOCHS):
            running_loss = 0.0
            running_correct = 0.0
            for data in tqdm(self.trainLoader):
                data, target = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterian(outputs, target)
                running_loss += loss 
                _, predictions = torch.max(outputs.data, 1)
                running_correct += (predictions == target).sum().item()

                loss.backward()
                self.optimizer.step()

            
            loss = running_loss / len(self.trainLoader.dataset)
            accuracy = 100. * running_correct / len(self.trainLoader.dataset)

            val_loss, val_accuracy = self.evaluate()

            print(f"Train accuracy: {accuracy:.2f}, Train loss: {loss:.4f}, "
                  f"Validation accuracy: {val_accuracy:.2f}, Validation loss: {val_loss:.4f}")
            
        print("Training complete!")


    def evaluate(self, validate=False):

        self.model.eval()
        val_loss = []
        val_accuracy = []

        dataloader = self.testLoader if not validate else self.valLoader

        with torch.no_grad():
            for batch in tqdm(dataloader):
                img, labels = batch[0].to(self.device), batch[1].to(self.device)

                logits = self.model(img)
                loss = self.criterian(logits, labels)
                val_loss.append(loss)

                preds = torch.argmax(logits, dim=1).flatten()
                accuracy = (preds == labels).cpu().numpy().mean() * 100
                val_accuracy.append(accuracy)

        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy
    

    def save_model_in_pth(self):
        model_path = self.config.MODEL_STORE_PATH
        print(f"saving model at {model_path}")
        torch.save(self.model.state_dcit(), model_path)

