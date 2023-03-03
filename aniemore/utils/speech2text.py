import whisper
from typing import NamedTuple, List
from functools import partial


class SpeechSegment(NamedTuple):
    """
    Структура для хранения результатов распознавания

    id: локальный номер сегмента
    seek: смещение в секундах от начала аудио
    start: время начала сегмента в секундах
    end: время конца сегмента в секундах
    text: распознанный текст
    tokens: список токенов
    temperature: whisper температура
    avg_logprob: средняя вероятность
    no_speech_prob: вероятность отсутствия речи
    compression_ratio: коэффициент сжатия
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
    """
    Структура для хранения результатов распознавания

    text: распознанный текст
    language: язык
    segments: список сегментов
    """
    text: str
    language: str
    segments: List[SpeechSegment]


class Speech2Text:
    def __init__(self, model_path: str):
        """
        Инициализация модели распознавания речи

        model_path: путь к модели
        """
        self.model = whisper.load_model(model_path)

    def __call__(self, audio_path: str) -> Speech2TextOutput:
        """
        Распознать аудио

        audio_path: путь к аудио

        return: результат распознавания

        >>> speech2text = Speech2Text('base')
        >>> speech2text('audio.wav')
        """
        return self.recognize(audio_path)

    def recognize(self, audio_path: str) -> Speech2TextOutput:
        """
        Распознать аудио

        audio_path: путь к аудио

        return: результат распознавания

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

