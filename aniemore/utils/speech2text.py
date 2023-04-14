import numpy
import torch
import whisper
from typing import NamedTuple, List, Union
from functools import partial


class SpeechSegment(NamedTuple):
    """Структура для хранения результатов распознавания

    Attributes:
     id(int): локальный номер сегмента
     seek(int): смещение в секундах от начала аудио
     start(float): время начала сегмента в секундах
     end(float): время конца сегмента в секундах
     text(str): распознанный текст
     tokens(List[int]): список токенов
     temperature(float): whisper температура
     avg_logprob(float): средняя вероятность
     no_speech_prob(float): вероятность отсутствия речи
     compression_ratio(float): коэффициент сжатия
    """
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float


class Speech2TextOutput(NamedTuple):
    """Структура для хранения результатов распознавания

    Attributes:
     text(str): распознанный текст
     language(str): язык
     segments(List[SpeechSegment]): список сегментов

    """
    text: str
    language: str
    segments: List[SpeechSegment]


class Speech2Text:
    def __init__(self, model_path: str):
        """Инициализация модели распознавания речи

        Args:
         model_path: путь к модели

        """
        self.model = whisper.load_model(model_path)

    def __call__(self, audio_path: str) -> Speech2TextOutput:
        """
        Распознать аудио

        Args:
         audio_path: путь к аудио

        Returns:
         результат распознавания

        >>> speech2text = Speech2Text('base')
        >>> speech2text('audio.wav')
        """
        return self.recognize(audio_path)

    def recognize(self, audio_path: Union[str, torch.Tensor, numpy.ndarray]) -> Speech2TextOutput:
        """Распознать аудио

        Args:
         audio_path: путь к аудио

        Returns:
          результат распознования

        Examples:
         >>> speech2text = Speech2Text('base')
         >>> speech2text.recognize('audio.wav')
        """
        result = self.model.transcribe(audio_path)
        result['segments'] = [SpeechSegment(**x) for x in result['segments']]
        return Speech2TextOutput(**result)


TinySpeech2Text = partial(Speech2Text, 'tiny')

BaseSpeech2Text = partial(Speech2Text, 'base')

SmallSpeech2Text = partial(Speech2Text, 'small')

MediumSpeech2Text = partial(Speech2Text, 'medium')

LargeSpeech2Text = partial(Speech2Text, 'large')

