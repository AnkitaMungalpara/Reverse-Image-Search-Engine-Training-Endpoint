import json
import os
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from src.entity.config_entity import EmbeddingConfig, ImageFolderConfig
from src.utils.database_handler import MongoDBClient

ImageRecord = namedtuple("ImageRecord", ["img", "label", "s3_link"])


class ImageFolder(Dataset):
    def __init__(self, label_map: Dict) -> None:
        self.config = ImageFolderConfig()
        self.config.LABEL_MAP = label_map
        self.transform = self.transformations()
        self.image_records: List[ImageRecord] = []
        self.record = ImageRecord

        file_list = os.listdir(self.config.ROOT_DIR)

        for class_path in file_list:
            path = os.path.join(self.config.ROOT_DIR, f"{class_path}")
            images = os.listdir(path)

            for image in tqdm(images):
                image_path = Path(f"""{self.config.ROOT_DIR}/{class_path}/{image}""")
                self.image_records.append(
                    self.record(
                        img=image_path,
                        label=self.config.LABEL_MAP[class_path],
                        s3_link=self.config.S3_LINK.format(
                            self.config.BUCKET, class_path, image
                        ),
                    )
                )

    def transformations(self):
        TRANSFORMED_IMG = transforms.Compose(
            [
                transforms.Resize(self.config.IMAGE_SIZE),
                transforms.CenterCrop(self.config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        return TRANSFORMED_IMG

    def __len__(self):
        return len(self.image_records)

    def __getitem__(self, index) -> Any:
        record = self.image_records[index]
        images, targets, links = record.img, record.label, record.s3_link
        images - Image.open(images)

        if len(images.getbands()) < 3:
            images = images.convert("RGB")
        images = np.array(self.transform(images))
        targets = torch.from_numpy(np.array(targets))
        images = torch.from_numpy(images)

        return images, targets, links


class EmbeddingGenerator:
    def __init__(self, model, device):
        self.config = EmbeddingConfig()
        self.mongo = MongoDBClient()
        self.model = model
        self.device = device
        self.embedding_model = self.load_model()
        self.embedding_model.eval()

    def load_model(self):
        model = self.model.to(self.device)
        model.load_state_dict(
            torch.load(self.config.MODEL_STORE_PATH, map_location=self.device)
        )

        return nn.Sequential(*list(model.children()))[:-1]

    def run_step(self, batch_size, image, label, s3_link):

        records = dict()

        images = self.embedding_model(image.to(self.device))
        images = images.detach().cpu().numpy()

        records["images"] = images.tolist()
        records["label"] = label.tolist()
        records["s3_link"] = s3_link

        df = pd.DataFrame(records)
        records = list(json.loads(df.T.to_json()).values())
        self.mongo.insert_bulk_records(records)

        return {"Response": f"Completed Generating Embeddings for {batch_size}."}
