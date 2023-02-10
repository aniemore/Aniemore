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
        self.MODEL_URL = model_url if model_url is not None else self.MODEL_URL
        self.device = device
        if setup_on_init:
            self.__setup_variables()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if value != 'cpu':
            if re.match(r'^(cuda)(:[0-9]+)?$', value) is None:  # https://regex101.com/r/SGEiYz/2
                raise ValueError(f"Device must be 'cpu' or 'cuda', or 'cuda:<number>', not {self.device}")
        self._device = value

    def __setup_variables(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_URL)
        self.model = BertForSequenceClassification.from_pretrained(self.MODEL_URL)
        self.model_config = BertConfig.from_pretrained(self.MODEL_URL)

    def _predict_one(self, text: str, single_label) -> Union[List[dict], List[str]]:
        """
        Получаем строку текста, токенизируем, отправляем в модель и возвращаем лист "эмоция : вероятность"

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

    def _predict_many(self, texts: List[str], single_label) -> Union[List[List[dict]], List[List[str]]]:
        """
        Он принимает список текстов и возвращает список прогнозов.

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

    def predict(self, text: Union[List[str], str], single_label=False) -> Union[List[dict], List[List[dict]],
    List[str], List[List[str]]]:
        """
        Эта функция принимает путь к файлу или список путей к файлам и возвращает список словарей или список списков
        словарей

        :param single_label: Вернуть наиболее вероятный класс или список классов с вероятностями
        :param text: Путь к изображению, которое вы хотите предсказать
        :type text: List[str] or str
        """
        if self.model is None:
            self.__setup_variables()

        if type(text) == str:
            return self._predict_one(text, single_label=single_label)

        elif type(text) == list:
            return self._predict_many(text, single_label=single_label)

        else:
            raise ValueError("You need to input list[text] or one text of your file for prediction")
