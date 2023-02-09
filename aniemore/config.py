import os
from dataclasses import dataclass

from aniemore.utils.custom_classes import AttributeDict


@dataclass
class Text:
    #  'str': int, - for code readability purpose
    labels = AttributeDict({
        'neutral':      0,
        'happiness':    1,
        'sadness':      2,
        'enthusiasm':   3,
        'fear':         4,
        'anger':        5,
        'disgust':      6,
    })

@dataclass
class Voice:
    #  'str': int, - for code readability purpose
    labels = AttributeDict({
        'anger':        0,
        'disgust':      1,
        'enthusiasm':   2,
        'fear':         3,
        'happiness':    4,
        'neutral':      5,
        'sadness':      6,
    })
    preprocess = AttributeDict({
        'audio-default-sample': 16000,
        'max-audio-duration-millis': 15000,
    })

@dataclass
class HuggingFace:
    models = AttributeDict({
        'wav2vec2_53_voice': 'aniemore/wav2vec2-xlsr-53-russian-emotion-recognition',
        'rubert_tiny2_text': 'aniemore/rubert-tiny2-russian-emotion-detection',
        'wav2vec2_53_asr': 'jonatasgrosman/wav2vec2-large-xlsr-53-russian',
    })

    datasets = AttributeDict({
        'resd': 'aniemore/resd',
        'cedr-m7': 'aniemore/cedr-m7',
    })

@dataclass
class YandexCloud:
    # set up your env variables like this:
    # -------------------------------------------------------------------
    #       YC_FOLDER_ID = YOUR_FOLDER_ID
    #       YC_IAM_TOKEN = YOUR_IAM_TOKEN (don't mind about 'created at')
    # -------------------------------------------------------------------
    YC_URL = 'https://stt.api.cloud.yandex.net/speech/v1/stt:recognize'
    YC_FOLDER_ID = os.getenv('YC_FOLDER_ID')
    YC_IAM_TOKEN = AttributeDict({
        'created_at': 1653322904.0637176,
        'token': os.getenv('YC_IAM_TOKEN'),
    })

