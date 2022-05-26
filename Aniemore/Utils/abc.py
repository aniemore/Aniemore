from abc import abstractmethod


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
        pass

    def to(self, a):
        """
        Это ничего не делает.

        :param a: Адрес пункта назначения
        """
        pass

    def setup_variables(self):
        """

        """
        pass

    def _predict_one(self, a):
        """


        :param a: список входных значений
        """
        pass

    def _predict_many(self, a):
        """
        > Эта функция принимает список чисел и возвращает список чисел

        :param a: Входные данные
        """
        pass

    def predict(self, a):
        """
        Функция, которая принимает параметр a и ничего не возвращает.

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

