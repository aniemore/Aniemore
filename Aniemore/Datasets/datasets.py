import datasets
from Aniemore.Utils import MasterDataset
from Aniemore.config import config


class Resd(MasterDataset):
    DATASET_URL = config["Huggingface"]["datasets"]["resd"]

    def __init__(self):
        pass

    def setup(self):
        self.loaded_dataset = datasets.load_dataset(self.DATASET_URL)
