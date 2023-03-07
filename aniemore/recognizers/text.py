"""
Модуль для распознавания эмоций в тексте
"""
import sys
from typing import List, Union, Tuple, Any
import torch
import warnings
import torch.nn.functional as F
import numpy as np
from transformers import  AutoTokenizer

from aniemore.utils.classes import BaseRecognizer


class TextRecognizer(BaseRecognizer):
    """
    Используем уже обученную (на модифированном CEDR датасете) rubert-tiny2 модель.
    Список эмоций и их ID в модели можете посмотроеть в config.yml
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_torch_scores(
            self,
            text: Union[str, List[str]],
            tokenizer: AutoTokenizer,
            device: str,
            max_length: int = 512,
            padding: bool = True,
            truncation: bool = True) -> torch.Tensor:
        """
        [PRIVATE METHOD] Получаем лист текстов, токенизируем, отправляем в модель и возвращаем тензор с вероятностями
        :param text: текст для анализа
        :type text: str
        :param tokenizer: токенайзер
        :type tokenizer: AutoTokenizer
        :param device: 'cpu' or 'cuda' or 'cuda:<number>'
        :type device: str
        :param max_length: максимальная длина текста (default=512)
        :type max_length: int
        :param padding: если True, то добавляем паддинги (default=True)
        :type padding: bool
        :param truncation: если True, то обрезаем текст (default=True)
        :type truncation: bool
        :return: torch.Tensor
        """
        inputs = tokenizer(text, max_length=max_length, padding=padding,
                                truncation=truncation, return_tensors='pt').to(device)
        with torch.no_grad():
            logits = self._model.to(self.device)(**inputs).logits
        scores = F.softmax(logits, dim=1)
        return scores

    # def _setup_variables(self) -> None:
    #     """
    #     [PRIVATE METHOD] Загружаем модель и токенайзер в память
    #
    #     :return: None
    #     """
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_URL)
    #     self.model = BertForSequenceClassification.from_pretrained(self.MODEL_URL)
    #     self.model_config = BertConfig.from_pretrained(self.MODEL_URL)

    def _predict_one(self, text: str, single_label: bool) -> Union[Tuple[dict], Tuple[str]]:
        """
        [PRIVATE METHOD] Получаем строку текста, токенизируем, отправляем в модель и возвращаем лист "эмоция : вероятность"

        :param text: текст для анализа
        :type text: str
        :return: список "эмоция : вероятность"
        """

        scores = self._get_torch_scores(text, self.tokenizer, self.device)

        if single_label:
            max_score = torch.argmax(scores, dim=1).numpy()
            outputs = [self.config.id2label[max_score[0]]]

        else:
            scores = scores[0]
            outputs = [{self.config.id2label[i]: v for i, v in enumerate(scores)}]

        return tuple(outputs)

    def _predict_many(self, texts: List[str], single_label: bool) -> \
            tuple[list[str | Any] | list[str | dict], ...]:
        """
        [PRIVATE METHOD] Он принимает список текстов и возвращает список прогнозов.

        :param texts: Список[стр]
        :type texts: List[str]
        :param single_label: Если True, функция вернет список строк. Если False, он вернет список словарей
        """
        scores = self._get_torch_scores(texts, self.tokenizer, self.device).detach().cpu().numpy()
        outputs = []

        for _text, _local_score in zip(texts, scores):
            if single_label:
                max_score: Union[str, Any] = np.argmax(_local_score)
                outputs.append(
                    [_text, self.config.id2label[max_score]]
                )
            else:
                outputs.append(
                    [_text, {self.config.id2label[i]: v for i, v in enumerate(_local_score)}]
                )
        print(single_label, outputs, type(outputs))
        return tuple(outputs)

    def predict(self, text: Union[List[str], str], single_label=False) -> Any:
        """
        Эта функция принимает путь к файлу или список путей к файлам и возвращает список словарей или список списков
        словарей

        :param single_label: Вернуть наиболее вероятный класс или список классов с вероятностями
        :param text: Путь к изображению, которое вы хотите предсказать
        :type text: List[str] or str
        """
        if self._model is None:
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
        if sys.platform == 'darwin':  # MacOS check
            warning_text = ("Silero models are not supported on MacOS. "
                            "To make it work, we've changed torch engine to `qnnpack`. Use this with caution.")
            warnings.warn(warning_text, category=UserWarning)
            torch.backends.quantized.engine = 'qnnpack'
        if self._grammar_model is None and self._apply_te is None:
            self._grammar_model, _, _, _, self._apply_te = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                          model='silero_te')

    def enhance(self, text: str) -> str:
        """
        Улучшение текста (исправление грамматических ошибок и т.д.)
        :param text: Текст, который нужно улучшить
        :return: Улучшенный текст
        """
        return self._apply_te(text.lower(), lan='ru')
