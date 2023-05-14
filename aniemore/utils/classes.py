"""
Custom classes for aniemore
"""
import torch
import gc
from contextlib import contextmanager
from typing import ClassVar, ContextManager, Any, List, Union, NamedTuple, Dict, Type

import transformers
from transformers import (
    PretrainedConfig,
    AutoConfig,
    AutoTokenizer,
    AutoFeatureExtractor,
    BertForSequenceClassification,
    PreTrainedModel
)

from aniemore.custom.modeling_classificators import BaseMultiModalForSequenceBaseClassification
from aniemore.models import Model

RecognizerOutputOne: Type[Dict[str, float]] = dict


class RecognizerOutputTuple(NamedTuple):
    """Структура для хранения результатов распознавания"""
    key: str
    output: RecognizerOutputOne


RecognizerOutputMany: Type[Dict[str, RecognizerOutputOne]] = dict


class BaseRecognizer:
    # all examples of this class will be saved in this list
    CLASS_HANDLERS: ClassVar[List[Any]] = []

    def __init__(self, model: Model = None, device: str = 'cpu', setup_on_init: bool = True) -> None:
        """
        Инициализируем класс

        Args:
         model(aniemore.models.Model): название модели из `aniemore.custom.classes`
         device(str): 'cpu' or 'cuda' or 'cuda:<number>'
         setup_on_init(bool): если True, то сразу загружаем модель и токенайзер в память

        Examples:
         >>> from aniemore.models import HuggingFaceModel
         >>> from aniemore.recognizers.text import TextRecognizer
         >>> tr = TextRecognizer(HuggingFaceModel.Text.Bert_Tiny, device='cuda:0')
         >>> tr.recognize('Как же я люблю природу, она прекрасна!!)))')
        """
        self._model: Any = None
        self._device: Union[str, None] = None
        self.config: Type[PretrainedConfig] = None
        self.model_cls: Type[PreTrainedModel] = None
        self.model_url: Union[str, None] = None
        self._logistic_fct: Any = None

        self.model = model
        self.device = device
        self._add_to_class_handlers()
        if setup_on_init:
            self._setup_variables()

    def _setup_variables(self) -> None:
        """Загружаем модель и экстрактор признаков в память
        """
        # check if model_cls is child of BaseMultiModalForSequenceClassification
        if issubclass(self.model_cls, BaseMultiModalForSequenceBaseClassification):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_url)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_url)
        elif self.model_cls is BertForSequenceClassification:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_url)
        else:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_url)

        try:
            self.config = AutoConfig.from_pretrained(self.model_url)
            self._model = self.model_cls.from_pretrained(self.model_url, config=self.config)
        except (RuntimeError, ValueError) as exc:
            self.config = AutoConfig.from_pretrained(self.model_url, trust_remote_code=True)
            self._model = self.model_cls.from_pretrained(
                self.model_url, trust_remote_code=True, config=self.config
            )
        finally:
            self._model = self._model.to(self.device)

        # check problem type and set logistic function (sigmoid or softmax, else softmax)
        if self.config.problem_type == 'single_label_classification':
            self._logistic_fct = torch.softmax
        elif self.config.problem_type == 'multi_label_classification':
            self._logistic_fct = torch.sigmoid
        else:
            self._logistic_fct = torch.softmax

    # add example to the list
    def _add_to_class_handlers(self):
        """Добавляем экземпляр класса в список всех экземпляров этого класса"""
        self.CLASS_HANDLERS.append(self)

    # remove example from the list
    def _remove_from_class_handlers(self):
        """Удаляем экземпляр класса из списка всех экземпляров этого класса"""
        if self in self.CLASS_HANDLERS:
            self.CLASS_HANDLERS.remove(self)

    # get all examples that is not an existing example of this class
    def _get_class_handlers(self):
        """Получаем список всех экземпляров этого класса, кроме текущего"""
        return [handler for handler in self.CLASS_HANDLERS if handler is not self]

    @property
    def device(self) -> str:
        """Возвращаем устройство, на котором будет работать модель"""
        return self._device

    @device.setter
    def device(self, value) -> None:
        """Устанавливаем устройство, на котором будет работать модель

        Args:
          value: возможные значения: 'cpu', 'cuda', 'cuda:<number>'

        Returns:
          None or raises ValueError

        """
        if value != 'cpu':
            if not self.validate_device(value):
                raise ValueError(f"Device must be 'cpu' or 'cuda', 'cuda:<number>' or 'mps', not {value}")
        self._device = value

        # set model to the given device
        if self._model is not None:
            self._model.to(self.device)

            if value != 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()

    @classmethod
    def validate_device(cls, value) -> bool:
        """Валидатор для устройства, на котором будет работать модель

        Args:
          value: возможные значения: 'cpu', 'cuda', 'cuda:<number>'

        Returns:
          True or False

        """
        try:
            torch.device(value)
            return True
        except RuntimeError:  # torch device error
            return False
        # if value != 'cpu':
        #     if re.match(r'^(cuda)(:\d+)?$', value) is None:  # https://regex101.com/r/SGEiYz/2
        #         return False
        # return True

    @property
    def model(self) -> Model:
        """Возвращаем текущую модель, которая будет распозновать данные

        Returns:
            модель, которая загружена в текущий момент
        """
        return Model(model_cls=self.model_cls, model_url=self.model_url)

    @model.setter
    def model(self, model: Model) -> None:
        """Устанавливаем модель, которая будет распозновать данные

        Args:
          model: валидная модель (тип модели смотрите в `aniemore.config.Model`)

        Returns:
          None

        """
        if self.validate_model(model):
            self.model_cls, self.model_url = model
            self._model = self.model_cls.from_pretrained(self.model_url)
        else:
            raise ValueError('Not a valid model provided: %s', model)

    @classmethod
    def validate_model(cls, model: Model) -> bool:
        """
        Валидатор для загружаемой модели
        Args:
          model: модель из `models.py`

        Returns:
         `True` если прошел валидацию, `False` если не прошёл
        """
        return all([
            isinstance(model, Model),
            isinstance(model.model_url, str),
            issubclass(model.model_cls, transformers.PreTrainedModel),
        ])

    # create a context manager that allows this proof of work:
    @contextmanager
    def on_device(
            self,
            device: Union[str, torch.device],
            clear_same_device_cache: bool = True,
            clear_cache_after: bool = True) -> ContextManager:
        """Context manager that allows you to switch the model to the given device

        Args:
          device: cpu' or 'cuda' or 'cuda:<number>'
          clear_same_device_cache: clear cuda cache after switching to the given device
          clear_cache_after: clear cuda cache after switching to the original device

        Returns:
          None

        Examples:
         >>> with model.on_device('cuda'):
         >>>     # do something
        """
        try:
            # get other examples of this class and switch them to cpu device
            for handler in self._get_class_handlers():
                # check if the device is already the same
                if handler.device == device and clear_same_device_cache:
                    # move to cpu
                    handler._model = handler._model.to('cpu')

            # switch this example to the given device
            self._model = self._model.to(device)

            # clear cuda cache
            if clear_same_device_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()

            yield
        finally:
            # switch this example to original device
            self._model = self._model.to(self._device)

            # clear cuda cache
            if clear_cache_after and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # get other examples of this class and switch them to their original device
            for handler in self._get_class_handlers():
                handler._model = handler._model.to(handler.device)

            # do garbage collection
            if clear_cache_after:
                gc.collect()

    @contextmanager
    def with_model(
            self,
            model: Model,
            device: Union[str, torch.device],
            clear_cache_after: bool = True) -> ContextManager:
        """Context manager that allows you to switch the model to the given model

        Args:
          model: model
          device: cpu' or 'cuda' or 'cuda:<number>'
          clear_cache_after: clear cuda cache after switching to the original device

        Returns:
          None

        Examples:
         >>> with vr.with_model(new_model, 'cuda') as new_model:
         >>>     # do something
        """
        new_handler = None

        try:
            # create new example of this class with the given model
            new_handler = self.__class__(model=model, device=device, setup_on_init=True)
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
        """[PROTECTED METHOD] Получаем тензор с предсказаниями модели

        Args:
          args: аргументы
          kwargs: аргументы

        Returns:
          тензор с предсказаниями модели

        """
        raise NotImplementedError

    def recognize(self, *args, **kwargs) -> Union[RecognizerOutputOne, RecognizerOutputMany]:
        """Получаем предсказания модели

        Args:
          args: аргументы
          kwargs: аргументы

        Returns:
          предсказания модели

        """
        raise NotImplementedError

    def _recognize_one(self, *args, **kwargs) -> RecognizerOutputOne:
        """[PROTECTED METHOD] Получаем предсказания модели для одного объекта

        Args:
          args: аргументы
          kwargs: аргументы

        Returns:
          предсказания модели

        """
        raise NotImplementedError

    def _recognize_many(self, *args, **kwargs) -> RecognizerOutputMany:
        """[PROTECTED METHOD] Получаем предсказания модели для нескольких объектов

        Args:
          args: аргументы
          kwargs: аргументы

        Returns:
          предсказания модели

        """
        raise NotImplementedError

    def _get_many_results(self, items: List[str], scores: torch.Tensor) -> RecognizerOutputMany:
        """[PROTECTED METHOD] Принимает на вход исследуемые объекты и результат от модели,
        и возвращает результат

        Args:
          items: список исследуемых объектов
          scores: выход от функции _get_torch_scores

        Returns:
          dict` с результатами по каждому объекту

        """
        result = []
        for path_, score in zip(items, scores):
            score = {k: v for k, v in zip(self.config.id2label.values(), score.tolist())}
            result.append(RecognizerOutputTuple(path_, RecognizerOutputOne(**score)))
        return RecognizerOutputMany(tuple(result))

    @classmethod
    def _get_single_label(
            cls,
            output: Union[RecognizerOutputOne, RecognizerOutputMany]
    ) -> Union[str, dict]:
        """[PROTECTED CLASS METHOD] Получаем метку из предсказаний модели

        Args:
          output: предсказания модели

        Returns:
          метка

        """
        # check if output is dict of [str: float]
        if isinstance(output, dict) and all(isinstance(x, float) for x in output.values()):
            # max score in dict
            return max(output, key=output.get)

        # check if output is dict of [str: dict[str: float]]
        if isinstance(output, dict) and all(isinstance(x, dict) for x in output.values()):
            # max score in list
            return {x: max(output[x], key=output[x].get) for x in output.keys()}

    @classmethod
    def _get_top_n_labels(
            cls,
            output: Union[RecognizerOutputOne, RecognizerOutputMany],
            n: int
    ) -> List[str] | dict:
        """[PROTECTED CLASS METHOD] Получаем метку из предсказаний модели

        Args:
            output: предсказания модели
            n: количество меток, которые нужно вернуть

        Returns:
            метка

        """
        # check if output is dict of [str: float]
        if isinstance(output, dict) and all(isinstance(x, float) for x in output.values()):
            # max score in dict
            return sorted(output, key=output.get, reverse=True)[:n]

        # check if output is dict of [str: dict[str: float]]
        if isinstance(output, dict) and all(isinstance(x, dict) for x in output.values()):
            # max score in list
            return {x: sorted(output[x], key=output[x].get, reverse=True)[:n] for x in output.keys()}

    def __del__(self):
        """
        Деструктор

        :return: None
        """
        self._remove_from_class_handlers()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
