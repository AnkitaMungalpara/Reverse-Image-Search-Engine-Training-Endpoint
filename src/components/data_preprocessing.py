from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.entity.config_entity import DataPreprocessingConfig


class DataProcessing:
    def __int__(self):
        self.config = DataPreprocessingConfig()

    def transformations(self):
        """
        PyTorch transfomations class for applyting transformations to an image
        """
        try:
            TRANSFORMED_IMG = transforms.Compose(
                [
                    transforms.Resize(self.config.IMAGE_SIZE),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            return TRANSFORMED_IMG
        except Exception as e:
            raise e

    def create_loaders(self, TRANSFORMED_IMG):
        try:
            print("Generatine Dataloaders...")

            for _ in tqdm(range(1)):
                train_data = ImageFolder(
                    root=self.config.TRAIN_DATA_PATH, transform=TRANSFORMED_IMG
                )
                test_data = ImageFolder(
                    root=self.config.TEST_DATA_PATH, transform=TRANSFORMED_IMG
                )
                val_data = ImageFolder(
                    root=self.config.VALID_DATA_PATH, transform=TRANSFORMED_IMG
                )

                train_data_loader = DataLoader(
                    train_data,
                    batch_size=self.config.BATCH_SIZE,
                    shuffle=True,
                    num_workers=1,
                )

                test_data_loader = DataLoader(
                    test_data,
                    batch_size=self.config.BATCH_SIZE,
                    shuffle=True,
                    num_workers=1,
                )

                val_data_loader = DataLoader(
                    val_data,
                    batch_size=self.config.BATCH_SIZE,
                    shuffle=True,
                    num_workers=1,
                )

                results = {
                    "train_data_loader": (train_data_loader, train_data),
                    "test_data_loader": (test_data_loader, test_data),
                    "val_data_loader": (val_data_loader, val_data),
                }

            return results

        except Exception as e:
            raise e

    def run_step(self):
        try:
            TRANSFORMED_IMG = self.transformations()
            results = self.create_loaders(TRANSFORMED_IMG)
            return results
        except Exception as e:
            raise e
