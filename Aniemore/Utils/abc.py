from abc import abstractmethod
import torch


class MasterModel:
    model_config = None
    feature_extractor = None
    processor = None
    tokenizer = None
    model = None
    device = None

    def __init__(self):
        """
        Конструктор.
        """
        self.device = torch.device("cpu")

    def to(self, device):
        """
        Если устройство является строкой, оно будет преобразовано в объект torch.device. Если устройство является объектом
        torch.device, то ему будет присвоен атрибут self.device. Если устройство не является ни строкой, ни объектом
        torch.device, будет выдано сообщение об ошибке.

        :param device: Устройство для запуска модели
        :return: Сам класс.
        """
        if type(device) == str:
            self.device = {
                "cuda": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                "gpu": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                "cpu": torch.device("cpu")
            }.get(device, torch.device("cpu"))
            return self

        elif type(device) == torch.device:
            self.device = device
            return self

        else:
            raise ValueError("Unknown acceleration device")

    def setup_variables(self):
        """

        """
        pass

    def _predict_one(self, a, single_label):
        """


        :param a: список входных значений
        """
        pass

    def _predict_many(self, a, single_label):
        """
        > Эта функция принимает список чисел и возвращает список чисел

        :param a: Входные данные
        """
        pass

    def predict(self, a, single_label=False):
        """
        Функция, которая принимает параметр a и ничего не возвращает.

        :param single_label: Вернуть единственный предсказанный класс или вероятности всех классов
        :param a: Вход в сеть
        """
        pass


class MasterDataset:
    loaded_dataset = None

    def setup(self):
        ...

    def __getitem__(self, key):
        """
        Возвращает значение ключа в словаре

        :param key: Ключ для поиска в словаре
        """
        if self.loaded_dataset is None:
            self.setup()

        return self.loaded_dataset[key]

