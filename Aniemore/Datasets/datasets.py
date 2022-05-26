import datasets
from Aniemore.Utils import MasterDataset
from Aniemore.config import config


# > Этот класс загружает набор данных, токенизирует его и бинаризирует метки
class Cedr(MasterDataset):
    DATASET_URL = config["Huggingface"]["datasets"]["cedr-m7"]

    def __init__(self):
        pass

    def setup(self):
        """
        > `setup()` вызывается один раз в начале программы
        """
        self.loaded_dataset = datasets.load_dataset(self.DATASET_URL)

    @staticmethod
    def binarize_labels(x, labels):
        """
        Он принимает список меток и возвращает список двоичных меток.

        :param x: входные данные
        :param labels: список ярлыков
        :return: Список нулей с длиной количества меток.
        """
        a = [0 for _ in range(len(labels))]
        for i in x:
            a[i] = 1
        return a

    @staticmethod
    def tokenize_and_labels(self, dataset: datasets.Dataset, tokenizer, labels):
        """
        Он берет набор данных, токенизирует его, а затем преобразует метки в двоичный формат.

        :param dataset: Набор данных для токенизации
        :type dataset: datasets.Dataset
        :param tokenizer: Используемый токенизатор
        :param labels: Список меток, которые мы хотим предсказать
        :return: Набор данных с токенизированным текстом и бинарными метками.
        """
        return dataset.map(
            lambda x: tokenizer(x['text'], truncation=True), batched=True
        ).map(
            lambda x: {
                'label': [
                    float(y) for y in self.binarize_labels(eval(x["label2ids"]), labels)
                ]
            }, batched=False, remove_columns=['text', 'labels', 'source', 'label2ids']
        )


# > Этот класс является подклассом класса MasterDataset, и его цель — загрузить набор данных Resd из Huggingface.
class Resd(MasterDataset):
    DATASET_URL = config["Huggingface"]["datasets"]["resd"]

    def __init__(self):
        pass

    def setup(self):
        """
        > Функция `load_dataset` принимает URL-адрес в качестве аргумента и возвращает словарь данных
        """
        self.loaded_dataset = datasets.load_dataset(self.DATASET_URL)
