import os
import tarfile

from boto3 import Session

from src.entity.config_entity import s3Config


class S3Connector(object):
    def __init__(self):
        self.config = s3Config()
        self.session = Session(
            aws_access_key_id=self.config.ACCESS_KEY_ID,
            aws_secret_access_key=self.config.SECRET_KEY,
            region_name=self.config.REGION_NAME,
        )
        self.client = self.session.client("s3")
        self.s3 = self.session.resource("s3")
        self.bucket = self.s3.Bucket(self.config.BUCKET_NAME)

    def zip_files(self):
        folder = tarfile.open(self.config.ZIP_NAME, "w:gz")
        for path, name in self.config.ZIP_PATHS:
            folder.add(path, name)
        folder.close()

        self.s3.meta.client.upload_file(
            self.config.ZIP_NAME,
            self.config.BUCKET_NAME,
            f"{self.config.KEY}/{self.config.ZIP_NAME}",
        )

        os.remove(self.config.ZIP_NAME)

    def pull_artifacts(self):
        self.bucket.download_file(
            f"{self.config.KEY}/{self.config.ZIP_NAME}", self.config.ZIP_NAME
        )
        dir = tarfile.open(self.config.ZIP_NAME)
        dir.extractall()
        dir.close()
        os.remove(self.config.ZIP_NAME)


if __name__ == "__main__":
    connect = S3Connector()
