"""
Custom classes for aniemore
"""
import torch
import gc
import re
from contextlib import contextmanager
from typing import ClassVar, ContextManager, Any, List, Union, TypeAlias, NamedTuple


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


ModelOutput: TypeAlias = dict[str, float]


class RecognizerOutput(NamedTuple):
    """
    Структура для хранения результатов распознавания
    """
    filename: str
    output: ModelOutput


RecognizerOutputRepr: TypeAlias = dict[str, ModelOutput]


class BaseRecognizer:
    # all examples of this class will be saved in this list
    CLASS_HANDLERS: ClassVar[List[Any]] = []
    model: Any
    _device: str

    def __init__(self, device: str, setup_on_init: bool = True, **kwargs) -> None:
        self.device = device
        self._add_to_class_handlers()

        if setup_on_init:
            self._setup_variables()

    # add example to the list with weak references
    def _add_to_class_handlers(self):
        self.CLASS_HANDLERS.append(self)

    # remove example from the list
    def _remove_from_class_handlers(self):
        if self in self.CLASS_HANDLERS:
            self.CLASS_HANDLERS.remove(self)

    # get all examples that is not an existing example of this class
    def _get_class_handlers(self):
        return [handler for handler in self.CLASS_HANDLERS if handler is not self]

    def _setup_variables(self, *args, **kwargs):
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

    def _get_torch_scores(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def _predict_one(self, *args, **kwargs):
        raise NotImplementedError

    def _predict_many(self, *args, **kwargs):
        raise NotImplementedError

    def __del__(self):
        self._remove_from_class_handlers()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
