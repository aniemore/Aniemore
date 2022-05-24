import datasets
from Aniemore.Utils import MasterDataset
from Aniemore.config import config


class Cedr(MasterDataset):
    DATASET_URL = config["Huggingface"]["datasets"]["cedr-m7"]

    def __init__(self):
        pass

    def setup(self):
        self.loaded_dataset = datasets.load_dataset(self.DATASET_URL)

    @staticmethod
    def binarize_labels(x, labels):
        a = [0 for _ in range(len(labels))]
        for i in x:
            a[i] = 1
        return a

    @staticmethod
    def tokenize_and_labels(self, dataset: datasets.Dataset, tokenizer, labels):
        return dataset.map(
            lambda x: tokenizer(x['text'], truncation=True), batched=True
        ).map(
            lambda x: {
                'label': [
                    float(y) for y in self.binarize_labels(eval(x["label2ids"]), labels)
                ]
            }, batched=False, remove_columns=['text', 'labels', 'source', 'label2ids']
        )


class Resd(MasterDataset):
    DATASET_URL = config["Huggingface"]["datasets"]["resd"]

    def __init__(self):
        pass

    def setup(self):
        self.loaded_dataset = datasets.load_dataset(self.DATASET_URL)

    def resample(self):
        pass
