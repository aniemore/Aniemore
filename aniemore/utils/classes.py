"""
Custom classes for aniemore
"""
import torch
import gc
import re
from contextlib import contextmanager
from typing import ClassVar, ContextManager, Any, List, Union, TypeAlias, NamedTuple
from aniemore.config import Model

RecognizerOutputOne: TypeAlias = dict[str, float]


class RecognizerOutputTuple(NamedTuple):
    """
    Структура для хранения результатов распознавания
    """
    key: str
    output: RecognizerOutputOne


RecognizerOutputMany: TypeAlias = dict[str, RecognizerOutputOne]


class BaseRecognizer:
    # all examples of this class will be saved in this list
    CLASS_HANDLERS: ClassVar[List[Any]] = []
    model: Any
    config: Any
    _device: str

    def __init__(self, device: str, setup_on_init: bool = True, *args, **kwargs) -> None:
        """
        Инициализируем класс

        :param device: 'cpu' or 'cuda' or 'cuda:<number>'
        :param setup_on_init: если True, то сразу загружаем модель и токенайзер в память

        :param args: аргументы для инициализации класса
        :param kwargs: аргументы для инициализации класса
        """
        self.device = device
        self._add_to_class_handlers()

        if setup_on_init:
            self._setup_variables()

    # add example to the list
    def _add_to_class_handlers(self):
        """
        Добавляем экземпляр класса в список всех экземпляров этого класса
        :return: None
        """
        self.CLASS_HANDLERS.append(self)

    # remove example from the list
    def _remove_from_class_handlers(self):
        """
        Удаляем экземпляр класса из списка всех экземпляров этого класса
        :return: None
        """
        if self in self.CLASS_HANDLERS:
            self.CLASS_HANDLERS.remove(self)

    # get all examples that is not an existing example of this class
    def _get_class_handlers(self):
        """
        Получаем список всех экземпляров этого класса, кроме текущего
        :return: List[Any]
        """
        return [handler for handler in self.CLASS_HANDLERS if handler is not self]

    def _setup_variables(self, *args, **kwargs):
        """
        Устанавливаем переменные
        :return: None
        """
        raise NotImplementedError

    @property
    def device(self) -> str:
        """
        Возвращаем устройство, на котором будет работать модель

        :return: 'cpu' or 'cuda' or 'cuda:<number>'
        """
        return self._device

    @device.setter
    def device(self, value) -> None:
        """
        Устанавливаем устройство, на котором будет работать модель

        :param value: возможные значения: 'cpu', 'cuda', 'cuda:<number>'
        :return: None or raises ValueError
        """
        if value != 'cpu':
            if not self.device_validator(value):
                raise ValueError(f"Device must be 'cpu' or 'cuda', or 'cuda:<number>', not {self.device}")
        self._device = value

        # set model to the given device
        if self.model is not None:
            self.model.to(self._device)

            if value != 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def device_validator(value) -> bool:
        """
        Валидатор для устройства, на котором будет работать модель

        :param value: возможные значения: 'cpu', 'cuda', 'cuda:<number>'
        :return: True or False
        """
        if value != 'cpu':
            if re.match(r'^(cuda)(:\d+)?$', value) is None:  # https://regex101.com/r/SGEiYz/2
                return False

        return True

    # create a context manager that allows this proof of work:
    @contextmanager
    def on_device(self, device: Union[str, torch.device], clear_same_device_cache: bool = True,
                  clear_cache_after: bool = True) -> ContextManager:
        """
        Context manager that allows you to switch the model to the given device

        :param device: 'cpu' or 'cuda' or 'cuda:<number>'
        :param clear_same_device_cache: clear cuda cache after switching to the given device
        :param clear_cache_after: clear cuda cache after switching to the original device
        :return: None

        >>> with model.on_device('cuda'):
        >>>     # do something
        """
        try:
            # get other examples of this class and switch them to cpu device
            for handler in self._get_class_handlers():
                # check if the device is already the same
                if handler.device == device and clear_same_device_cache:
                    # move to cpu
                    handler.model = handler.model.to('cpu')

            # switch this example to the given device
            self.model = self.model.to(device)

            # clear cuda cache
            if clear_same_device_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()

            yield
        finally:
            # switch this example to original device
            self.model = self.model.to(self.device)

            # clear cuda cache
            if clear_cache_after and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # get other examples of this class and switch them to their original device
            for handler in self._get_class_handlers():
                handler.model = handler.model.to(handler.device)

            # do garbage collection
            if clear_cache_after:
                gc.collect()

    @contextmanager
    def with_model(self, model_name: Model, device: Union[str, torch.device],
                   clear_cache_after: bool = True) -> ContextManager:
        """
        Context manager that allows you to switch the model to the given model

        :param model_name: model
        :param device: 'cpu' or 'cuda' or 'cuda:<number>'
        :param clear_cache_after: clear cuda cache after switching to the original device
        :return: None

        >>> with vr.with_model(new_model, 'cuda') as new_model:
        >>>     # do something
        """
        new_handler = None

        try:
            # create new example of this class with the given model
            new_handler = self.__class__(model_name=model_name, device=device, setup_on_init=True)

            yield new_handler

        finally:
            # delete new example
            del new_handler

            # clear cuda cache
            if clear_cache_after and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # do garbage collection
            if clear_cache_after:
                gc.collect()

    def _get_torch_scores(self, *args, **kwargs) -> torch.Tensor:
        """
        Получаем тензор с предсказаниями модели

        :param args: аргументы
        :param kwargs: аргументы
        :return: тензор с предсказаниями модели
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> Union[RecognizerOutputOne, RecognizerOutputMany]:
        """
        Получаем предсказания модели

        :param args: аргументы
        :param kwargs: аргументы
        :return: предсказания модели
        """
        raise NotImplementedError

    def _predict_one(self, *args, **kwargs) -> RecognizerOutputOne:
        """
        Получаем предсказания модели для одного объекта

        :param args: аргументы
        :param kwargs: аргументы
        :return: предсказания модели
        """
        raise NotImplementedError

    def _predict_many(self, *args, **kwargs) -> RecognizerOutputMany:
        """
        Получаем предсказания модели для нескольких объектов

        :param args: аргументы
        :param kwargs: аргументы
        :return: предсказания модели
        """
        raise NotImplementedError

    @staticmethod
    def _get_single_label(output: Union[RecognizerOutputOne, RecognizerOutputMany]) -> \
            Union[str, dict]:
        """
        Получаем метку из предсказаний модели

        :param output: предсказания модели
        :return: метка
        """
        # check if output is dict of [str: float]
        if isinstance(output, dict) and all(isinstance(x, float) for x in output.values()):
            # max score in dict
            return max(output, key=output.get)

        # check if output is dict of [str: dict[str: float]]
        if isinstance(output, dict) and all(isinstance(x, dict) for x in output.values()):
            # max score in list
            return {x: max(output[x], key=output[x].get) for x in output.keys()}

    def __del__(self):
        """
        Деструктор

        :return: None
        """
        self._remove_from_class_handlers()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
