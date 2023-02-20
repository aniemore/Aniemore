from typing import Union, List

from aniemore.utils.classes import (
    BaseRecognizer,
    ModelOutput,
    RecognizerOutput,
    RecognizerOutputTuple
)
from aniemore.config_enums import HuggingFaceModel
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification


class VoiceRecognizer(BaseRecognizer):
    feature_extractor: AutoFeatureExtractor = None
    model: AutoModelForSequenceClassification = None

    def __init__(self, model_name: HuggingFaceModel, device: str = 'cpu', setup_on_init: bool = True) -> None:
        """
        Инициализируем класс
        :param model_name: одна из моделей из config.py
        :param device: 'cpu' or 'cuda' or 'cuda:<number>'
        :param setup_on_init: если True, то сразу загружаем модель и токенайзер в память
        """
        self.MODEL_CLS, self.MODEL_URL = model_name
        super().__init__(setup_on_init=setup_on_init, device=device)

    def _setup_variables(self) -> None:
        """
        Загружаем модель и экстрактор признаков в память
        :return: None
        """
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.MODEL_URL)
        try:
            self.model = self.MODEL_CLS.from_pretrained(self.MODEL_URL)
        except Exception:
            self.model = self.MODEL_CLS.from_pretrained(
                self.MODEL_URL, trust_remote_code=True
            )
        finally:
            self.model.to(self.device)

    @staticmethod
    def speech_file_to_array_fn(path):
        """
        Он берет путь к файлу .wav, считывает его и возвращает пустой массив аудиоданных.
        :param path: путь к файлу
        """
        speech_array, _sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    # TODO: Изменить название метода на _get_model_output
    def _get_torch_scores(self, speech: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Получаем выход модели
        :param speech: тензор аудиоданных
        :return: выход модели
        """
        sampling_rate = self.feature_extractor.sampling_rate
        inputs = self.feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to(self.model.device) for key in inputs}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # move inputs to cpu back
        for key in inputs:
            inputs[key] = inputs[key].cpu()

        del inputs

        scores = torch.softmax(logits, dim=1)
        return scores

    def _predict_one(self, path: str) -> ModelOutput:
        """
        Прогнозируем один файл
        :param path: путь к файлу
        :return: выход модели
        """
        speech = self.speech_file_to_array_fn(path)
        scores = self._get_torch_scores(speech)

        scores = {k: v for k, v in zip(self.model.config.id2label.values(), scores[0].tolist())}

        return ModelOutput(**scores)

    def _predict_many(self, paths: List[str]) -> RecognizerOutput:
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
            score = {k: v for k, v in zip(self.model.config.id2label.values(), score.tolist())}
            result.append(RecognizerOutputTuple(path_, ModelOutput(**score)))

        return RecognizerOutput(tuple(result))

    def predict(self, paths: Union[List[str], str]) -> Union[ModelOutput, RecognizerOutput]:
        """
        Прогнозируем файлы
        :param paths: путь к файлу или список путей к файлам
        :return: выход модели
        """
        if isinstance(paths, str):
            return self._predict_one(paths)
        elif isinstance(paths, list):
            return self._predict_many(paths)
        else:
            raise ValueError('paths must be str or list')
