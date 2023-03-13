from typing import Union, List
import numpy
import torch
import torchaudio

from aniemore.utils.classes import (
    BaseRecognizer,
    RecognizerOutputOne,
    RecognizerOutputMany
)


# noinspection PyUnresolvedReferences
class VoiceRecognizer(BaseRecognizer):
    @classmethod
    def speech_file_to_array_fn(cls, path):
        """
        Загружаем аудиофайл в массив
        :param path: путь к файлу
        :return: numpy.ndarray
        """
        speech_array, _sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def _get_torch_scores(self, speech: Union[List[torch.Tensor], torch.Tensor, numpy.ndarray]) -> torch.Tensor:
        """
        Получаем выход модели
        :param speech: ndarray
        :return: torch.Tensor
        """
        sampling_rate = self.feature_extractor.sampling_rate
        inputs = self.feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to(self._model.device) for key in inputs}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        # move inputs to cpu back
        for key in inputs:
            inputs[key] = inputs[key].cpu()

        del inputs

        scores = torch.softmax(logits, dim=1)
        return scores

    def _predict_one(self, path: str) -> RecognizerOutputOne:
        """
        Исполнение рантайма для одного файла
        :param path: путь к файлу
        :return: RecognizerOutputOne
        """
        speech = self.speech_file_to_array_fn(path)
        scores = self._get_torch_scores(speech)

        scores = {k: v for k, v in zip(self.config.id2label.values(), scores[0].tolist())}

        return RecognizerOutputOne(**scores)

    def _predict_many(self, paths: List[str]) -> RecognizerOutputMany:
        """
        Прогнозируем несколько файлов
        :param paths: список путей к файлам
        :return: словарь с выходами модели
        """

        speeches = []
        for path in paths:
            speech = self.speech_file_to_array_fn(path)
            speeches.append(speech)

        scores = self._get_torch_scores(speeches)

        result = []

        for path_, score in zip(paths, scores):
            score = {k: v for k, v in zip(self.config.id2label.values(), score.tolist())}
            result.append(RecognizerOutputTuple(path_, RecognizerOutputOne(**score)))

        return RecognizerOutputMany(tuple(result))

    # TODO: add single_label option

    def predict(self, paths: Union[List[str], str], return_single_label: bool = False) -> \
            Union[RecognizerOutputOne, RecognizerOutputMany]:
        """
        Прогнозируем файлы
        :param paths: путь к файлу или список путей к файлам
        :param return_single_label: если True, то возвращаем только один лейбл
        :return: выход модели
        """
        if isinstance(paths, str):
            if return_single_label:
                return self._get_single_label(self._predict_one(paths))

            return self._predict_one(paths)
        elif isinstance(paths, list):
            if return_single_label:
                return self._get_single_label(self._predict_many(paths))

            return self._predict_many(paths)
        else:
            raise ValueError('paths must be str or list')
