import os

import from_root
import splitfolders
from from_root import from_root

from src.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def download_data(self):
        """
        - prefix: prefix pattern to match in S3
        - local: local path where data is stored
        - bucket: S3 container with target content
        - client: initialize S3 client object


        """
        try:
            print(
                "--------------------------Fetaching data process started...---------------------------"
            )
            data_path = os.path.join(from_root(), self.config.RAW, self.config.PREFIX)
            os.system(
                f"aws s3 sync s3://image-database-system-01/images/ {data_path} --no-progress"
            )
            print(
                "--------------------------Fetaching data process completed...---------------------------"
            )
        except Exception as e:
            raise e

    def split_data(self):
        try:
            splitfolders.ratio(
                input=os.path.join(self.config.RAW, self.config.PREFIX),
                output=self.config.SPLIT,
                seed=self.config.SEED,
                ratio=self.config.RATIO,
                group_prefix=None,
                move=False,
            )
        except Exception as e:
            raise e

    def run_step(self):
        self.download_data()
        self.split_data()

        return {"Response": "Data Ingestion COmpleted!"}


if __name__ == "__main__":
    di = DataIngestion()
    di.run_step()
