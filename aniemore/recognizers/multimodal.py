from typing import Union, List, Dict, Tuple
import numpy
import torch
import torchaudio

from aniemore.models import Model
from aniemore.utils.classes import (
    BaseRecognizer,
    RecognizerOutputOne,
    RecognizerOutputMany,
)

from transformers import PreTrainedTokenizerBase, BatchEncoding

from aniemore.utils.speech2text import Speech2Text


class VoiceTextRecognizer(BaseRecognizer):
    @classmethod
    def speech_file_to_array_fn(cls, path):
        """Загружаем аудиофайл в массив

        Args:
          path: путь к файлу

        Returns:
          numpy.ndarray

        """
        speech_array, _sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    @classmethod
    def tokenize(cls, text: Union[str, List[str]], tokenizer: PreTrainedTokenizerBase) -> BatchEncoding:
        """Токенизация текста

        Args:
          text: текст
          tokenizer: токенайзер

        Returns:
          список токенов

        """

        return tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

    def _get_torch_scores(
            self,
            speech: Union[List[torch.Tensor], torch.Tensor, numpy.ndarray],
            text_inputs: Union[Dict[str, torch.Tensor], BatchEncoding]) -> torch.Tensor:
        """Получаем выход модели

        Args:
          speech: ndarray
          text_tokens: ndarray

        Returns:
          torch.Tensor

        """

        # check if first dimension of batch of inputs is the same
        if isinstance(speech, list):
            if len(speech) != len(text_inputs["input_ids"]):
                raise ValueError("Batch size of speech and text inputs should be the same")

        sampling_rate = self.feature_extractor.sampling_rate
        audio_inputs = self.feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        audio_inputs = {key: audio_inputs[key].to(self._model.device) for key in audio_inputs}

        text_inputs = {key: text_inputs[key].to(self._model.device) for key in text_inputs}

        inputs = dict(
            input_values=audio_inputs["input_values"],
            audio_attention_mask=audio_inputs["attention_mask"],
            input_ids=text_inputs["input_ids"],
            token_type_ids=text_inputs["token_type_ids"],
            text_attention_mask=text_inputs["attention_mask"]
        )

        with torch.no_grad():
            logits = self._model(**inputs).logits

        # move inputs to cpu back
        for key in inputs:
            inputs[key] = inputs[key].cpu()

        del inputs

        if self._logistic_fct is torch.sigmoid:
            scores = self._logistic_fct(logits)
        elif self._logistic_fct is torch.softmax:
            scores = self._logistic_fct(logits, dim=1)
        else:
            raise ValueError('logistic_fct must be torch.sigmoid or torch.softmax')

        return scores

    def _recognize_one(self, path: str, text: str) -> RecognizerOutputOne:
        """Исполнение рантайма для одного файла

        Args:
          path: путь к файлу
          text: текст

        Returns:
          RecognizerOutputOne

        """
        speech = self.speech_file_to_array_fn(path)
        tokens = self.tokenize(text, self.tokenizer)
        scores = self._get_torch_scores(speech, tokens)

        scores = {k: v for k, v in zip(self.config.id2label.values(), scores[0].tolist())}

        return RecognizerOutputOne(**scores)

    def _recognize_many(self, voice_text_couple: List[Tuple[str, str]]) -> RecognizerOutputMany:
        """Прогнозируем несколько файлов

        Args:
          voice_text_couple: список путей к файлам

        Returns:
          словарь с выходами модели

        """
        paths = [path for path, _ in voice_text_couple]

        speeches = []
        tokens = []
        for path, text in voice_text_couple:
            speech = self.speech_file_to_array_fn(path)
            speeches.append(speech)
            tokens.append(text)

        tokens = self.tokenize(tokens, self.tokenizer)
        scores: torch.Tensor = self._get_torch_scores(speeches, tokens)
        results: RecognizerOutputMany = self._get_many_results(paths, scores)

        return results

    def recognize(
            self,
            paths_text_couple: Union[List[Tuple[str, str]], Tuple[str, str]],
            return_single_label: bool = False,
            top_n: Union[int, None] = None
    ) -> Union[RecognizerOutputOne, RecognizerOutputMany]:
        """Прогнозируем файлы

        Args:
          paths_text_couple: пары путь к файлу и текст (str, str) или список таких пар
          return_single_label: если True, то возвращаем только один лейбл
          top_n: количество лейблов, которые необходимо вернуть
        Returns:
          выход модели

        """
        if self._model is None:
            self._setup_variables()

        # check if config.problem_type is exists
        if self.config.problem_type is not None:
            if self.config.problem_type == 'multi_label_classification':
                return_single_label = False

        if isinstance(paths_text_couple, tuple):
            if return_single_label:
                return self._get_single_label(self._recognize_one(*paths_text_couple))

            elif top_n is not None:
                return self._get_top_n_labels(self._recognize_one(*paths_text_couple), top_n)

            return self._recognize_one(*paths_text_couple)

        elif isinstance(paths_text_couple, list):
            if return_single_label:
                return self._get_single_label(self._recognize_many(paths_text_couple))

            elif top_n is not None:
                return self._get_top_n_labels(self._recognize_many(paths_text_couple), top_n)

            return self._recognize_many(paths_text_couple)

        else:
            raise ValueError('paths_text_couple should be tuple or list')


class MultiModalRecognizer(VoiceTextRecognizer):
    def __init__(
            self,
            model: Model = None,
            s2t_model: Speech2Text = None,
            device: str = 'cpu',
            setup_on_init: bool = True
    ) -> None:
        super().__init__(model, device, setup_on_init)
        self.s2t_model = s2t_model

    def speech_to_text(self, speech: Union[torch.Tensor, numpy.ndarray, str]) -> str:
        """Преобразование речи в текст

        Args:
          speech: ndarray

        Returns:
          текст

        """
        return self.s2t_model.recognize(speech).text.strip()

    def _recognize_one(self, path: str, **kwargs) -> RecognizerOutputOne:
        """Исполнение рантайма для одного файла

        Args:
          path: путь к файлу
          text: текст

        Returns:
          RecognizerOutputOne

        """
        text = self.speech_to_text(path)

        return super()._recognize_one(path, text)

    def _recognize_many(self, paths: List[str]) -> RecognizerOutputMany:
        """Прогнозируем несколько файлов

        Args:
          voice_text_couple: список путей к файлам

        Returns:
          словарь с выходами модели

        """

        couples = []

        for path in paths:
            text = self.speech_to_text(path)
            couples.append((path, text))

        return super()._recognize_many(couples)

    def recognize(
            self,
            paths: Union[List[str], str],
            return_single_label: bool = False,
            top_n: Union[int, None] = None
    ) -> Union[RecognizerOutputOne, RecognizerOutputMany]:
        """Прогнозируем файлы

        Args:
          paths: путь к файлу или список таких путей
          return_single_label: если True, то возвращаем только один лейбл
          top_n: количество лейблов, которые нужно вернуть

        Returns:
          выход модели

        """
        if self._model is None:
            self._setup_variables()

        # check if config.problem_type is exists
        if self.config.problem_type is not None:
            if self.config.problem_type == 'multi_label_classification':
                return_single_label = False

        if isinstance(paths, str):
            if return_single_label:
                return self._get_single_label(self._recognize_one(paths))

            elif top_n is not None:
                return self._get_top_n_labels(self._recognize_one(paths), top_n)

            return self._recognize_one(paths)

        elif isinstance(paths, list):
            if return_single_label:
                return self._get_single_label(self._recognize_many(paths))

            elif top_n is not None:
                return self._get_top_n_labels(self._recognize_many(paths), top_n)

            return self._recognize_many(paths)

        else:
            raise ValueError('paths should be str or list')
