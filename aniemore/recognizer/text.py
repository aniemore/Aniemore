import re
from typing import List, Union
import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertForSequenceClassification, AutoTokenizer, BertConfig

from aniemore import config


class TextRecognizer:
    """
    Используем уже обученную (на модифированном CEDR датасете) rubert-tiny2 модель.
    Список эмоций и их ID в модели можете посмотроеть в config.yml
    """
    MODEL_URL = config.HuggingFace.models.rubert_tiny2_text

    tokenizer: AutoTokenizer = None
    model: BertForSequenceClassification = None
    model_config: BertConfig = None
    _device: str = None

    def __init__(self, model_url: str = None, device: str = 'cpu', setup_on_init: bool = True) -> None:
        """
        Инициализируем класс
        :param model_url: одна из моделей из config.py
        :param device: 'cpu' or 'cuda' or 'cuda:<number>'
        :param setup_on_init: если True, то сразу загружаем модель и токенайзер в память
        """
        self.MODEL_URL = model_url if model_url is not None else self.MODEL_URL
        self.device = device
        if setup_on_init:
            self._setup_variables()

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
            if re.match(r'^(cuda)(:\d+)?$', value) is None:  # https://regex101.com/r/SGEiYz/2
                raise ValueError(f"Device must be 'cpu' or 'cuda', or 'cuda:<number>', not {self.device}")
        self._device = value

    def _setup_variables(self) -> None:
        """
        [PRIVATE METHOD] Загружаем модель и токенайзер в память

        :return: None
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_URL)
        self.model = BertForSequenceClassification.from_pretrained(self.MODEL_URL)
        self.model_config = BertConfig.from_pretrained(self.MODEL_URL)

    def _predict_one(self, text: str, single_label: bool) -> Union[List[dict], List[str]]:
        """
        [PRIVATE METHOD] Получаем строку текста, токенизируем, отправляем в модель и возвращаем лист "эмоция : вероятность"

        :param text: текст для анализа
        :type text: str
        :return: список "эмоция : вероятность"
        """
        inputs = self.tokenizer(text, max_length=512, padding=True,
                                truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            logits = self.model.to(self.device)(**inputs).logits

        scores = F.softmax(logits, dim=1)

        if single_label is False:
            scores = scores.numpy()[0]
            outputs = [{self.model_config.id2label[i]: v for i, v in enumerate(scores)}]

        else:
            max_score = torch.argmax(scores, dim=1).numpy()
            outputs = [self.model_config.id2label[max_score[0]]]

        return outputs

    def _predict_many(self, texts: List[str], single_label: bool) -> Union[List[List[dict]], List[List[str]]]:
        """
        [PRIVATE METHOD] Он принимает список текстов и возвращает список прогнозов.

        :param texts: Список[стр]
        :type texts: List[str]
        :param single_label: Если True, функция вернет список строк. Если False, он вернет список словарей
        """
        inputs = self.tokenizer(texts, max_length=512, padding=True,
                                truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            logits = self.model.to(self.device)(**inputs).logits

        scores = F.softmax(logits, dim=1).detach().cpu().numpy()

        outputs = []

        for _text, _local_score in zip(texts, scores):
            if single_label is False:
                outputs.append(
                    [_text, {self.model_config.id2label[i]: v for i, v in enumerate(_local_score)}]
                )

            else:
                max_score = np.argmax(_local_score)
                outputs.append(
                    [_text, self.model_config.id2label[max_score]]
                )

        return outputs

    def predict(self, text: Union[List[str], str], single_label=False) -> \
            Union[List[dict], List[List[dict]], List[str], List[List[str]]]:
        """
        Эта функция принимает путь к файлу или список путей к файлам и возвращает список словарей или список списков
        словарей

        :param single_label: Вернуть наиболее вероятный класс или список классов с вероятностями
        :param text: Путь к изображению, которое вы хотите предсказать
        :type text: List[str] or str
        """
        if self.model is None:
            self._setup_variables()

        if type(text) == str:
            return self._predict_one(text, single_label=single_label)
        elif type(text) == list:
            return self._predict_many(text, single_label=single_label)
        else:
            raise ValueError("You need to input list[text] or one text of your file for prediction")


class TextEnhancer:
    """
    Класс для улучшения текста, например, для исправления грамматических ошибок и т.д.
    """
    _grammar_model = None
    _apply_te = None

    def __init__(self, setup_on_init: bool = True) -> None:
        """
        Инициализация класса
        :param setup_on_init: Если True, модель будет загружена при инициализации класса
        """
        if setup_on_init:
            self.load_model()

    def load_model(self) -> None:
        """
        Загрузка модели. Если она уже загружена, то ничего не произойдет
        :return: None
        """
        if self._grammar_model is None and self._apply_te is None:
            self._grammar_model, _, _, _, self._apply_te = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                          model='silero_te')

    def enhance(self, text: str) -> str:
        """
        Улучшение текста (исправление грамматических ошибок и т.д.)
        :param text: Текст, который нужно улучшить
        :return: Улучшенный текст
        """
        return self._apply_te(text.lower(), lan='ru', model=self._grammar_model)
