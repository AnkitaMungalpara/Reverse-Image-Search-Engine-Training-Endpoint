import os

import torch
import tqdm
from from_root import from_root
from torch.utils.data import DataLoader

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataProcessing
from src.components.embeddings import EmbeddingGenerator, ImageFolder
from src.components.model import NeuralNet
from src.components.nearest_neighbours import Annoy
from src.components.trainer import Trainer
from src.utils.storage_handler import S3Connector


class Pipeline:
    def __init__(self):
        self.paths = [
            "data",
            "data/raw",
            "data/splitted",
            "data/embeddings",
            "model",
            "model/benchmark",
            "model/finetuned",
        ]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def initiate_data_ingestion(self):
        for folder in self.paths:
            path = os.path.join(from_root(), folder)
            if not os.path.exists(path):
                os.mkdir(folder)

            data_ingestion = DataIngestion()
            data_ingestion.run_step()


    @staticmethod
    def initiate_data_processing():
        dp = DataProcessing()
        loaders = dp.run_step()
        return loaders


    @staticmethod
    def initiate_model_architecture():
        return NeuralNet()


    def initiate_model_training(self, loaders, net):
        trainer = Trainer(loaders, self.device, net)
        trainer.train_model()
        trainer.evaluate(validate=True)
        trainer.save_model_in_pth()


    def generate_embeddings(self, loaders, net):
        data = ImageFolder(label_map=loaders["val_data_loader"][1].class_to_idx)
        data_loader = DataLoader(dataset=data, batch_size=64, shuffle=True)
        embeddings = EmbeddingGenerator(model=net, device=self.device)

        for batch, val in tqdm(enumerate(data_loader)):
            img, target, link = val
            print(embeddings.run_step(batch, img, target, link))


    @staticmethod
    def create_annoy():
        ann = Annoy()
        ann.run_step()


    @staticmethod
    def push_artifacts():
        connection = S3Connector()
        result = connection.zip_files()
        return result


    def run_pipeline(self):
        self.initiate_data_ingestion()
        loaders = self.initiate_data_processing()
        net = self.initiate_model_architecture()
        self.initiate_model_training(loaders=loaders, net=net)
        self.generate_embeddings(loaders=loaders, net=net)
        self.create_annoy()
        self.push_artifacts()

        return {"Respone": "Pipeline run completed!!!"}


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run_pipeline()
