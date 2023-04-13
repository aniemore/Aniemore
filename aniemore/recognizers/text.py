"""
Модуль для распознавания эмоций в тексте
"""
import sys
from typing import List, Union
import torch
import warnings
from transformers import PreTrainedTokenizerBase

from aniemore.utils.classes import (
    BaseRecognizer,
    RecognizerOutputMany,
    RecognizerOutputOne,
)


class TextRecognizer(BaseRecognizer):
    def _get_torch_scores(
            self,
            text: Union[str, List[str]],
            tokenizer: PreTrainedTokenizerBase,
            device: str,
            max_length: int = 512,
            padding: bool = True,
            truncation: bool = True) -> torch.Tensor:
        """[PROTECTED METHOD] Получаем лист текстов, токенизируем, отправляем в модель и возвращаем тензор с вероятностями

        Args:
          text: текст для анализа
          tokenizer: токенайзер
          device: cpu' or 'cuda' or 'cuda:<number>'
          max_length: максимальная длина текста (default=512)
          padding: если True, то добавляем паддинги (default=True)
          truncation: если True, то обрезаем текст (default=True)

        Returns:
          torch.Tensor

        """
        inputs = tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors='pt').to(device)

        with torch.no_grad():
            logits = self._model.to(self.device)(**inputs).logits

        if self._logistic_fct is torch.sigmoid:
            scores = self._logistic_fct(logits)
        elif self._logistic_fct is torch.softmax:
            scores = self._logistic_fct(logits, dim=1)
        else:
            raise ValueError('logistic_fct must be one of torch.sigmoid or torch.softmax')

        return scores

    def _recognize_one(self, text: str) -> RecognizerOutputOne:
        """[PROTECTED METHOD] Получаем строку текста, токенизируем, отправляем в модель и возвращаем "

        Args:
          text(str): текст для анализа

        Returns:
          результат распознования
        """

        scores = self._get_torch_scores(text, self.tokenizer, self.device)

        scores = {k: v for k, v in zip(self.config.id2label.values(), scores[0].tolist())}

        return RecognizerOutputOne(**scores)

    def _recognize_many(self, texts: List[str]) -> RecognizerOutputMany:
        """[PROTECTED METHOD] Принимает список текстов и возвращает список прогнозов.

        Args:
          texts: список текстов для анализа

        Returns:
          результат распознования
        """
        scores = self._get_torch_scores(texts, self.tokenizer, self.device).detach().cpu().numpy()
        results: RecognizerOutputMany = self._get_many_results(texts, scores)

        return results

    def recognize(self, text: Union[List[str], str], return_single_label=False) -> \
            Union[RecognizerOutputOne, RecognizerOutputMany]:

        if self._model is None:
            self._setup_variables()

        if isinstance(text, str):
            if return_single_label:
                return self._get_single_label(self._recognize_one(text))

            return self._recognize_one(text)
        elif isinstance(text, list):
            if return_single_label:
                return self._get_single_label(self._recognize_many(text))

            return self._recognize_many(text)
        else:
            raise ValueError('paths must be str or list')


class TextEnhancer:
    """Класс для улучшения текста, например, для исправления грамматических ошибок и т.д."""
    _grammar_model = None
    _apply_te = None

    def __init__(self, setup_on_init: bool = True) -> None:
        """Инициализация класса
        Args:
          setup_on_init: Если True, модель будет загружена при инициализации класса
        """
        if setup_on_init:
            self._load_model()

    def _load_model(self) -> None:
        """Загрузка модели. Если она уже загружена, то ничего не произойдет."""
        if sys.platform == 'darwin':  # MacOS check
            warning_text = ("Silero models are not supported on MacOS. "
                            "To make it work, we've changed torch engine to `qnnpack`. Use this with caution.")
            warnings.warn(warning_text, category=UserWarning)
            torch.backends.quantized.engine = 'qnnpack'
        if self._grammar_model is None and self._apply_te is None:
            self._grammar_model, _, _, _, self._apply_te = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                          model='silero_te')

    def enhance(self, text: str) -> str:
        """Улучшение текста (исправление грамматических ошибок и т.д.)

        Args:
          text: Текст, который нужно улучшить

        Returns:
          Улучшенный текст
        """
        return self._apply_te(text.lower(), lan='ru')
