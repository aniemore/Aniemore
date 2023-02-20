import os
from dataclasses import dataclass
from transformers import (
    Wav2Vec2ForSequenceClassification,
    WavLMForSequenceClassification,
    UniSpeechSatForSequenceClassification,
    HubertForSequenceClassification,
    AutoModelForSequenceClassification,
    BertForSequenceClassification
)

from aniemore.utils.classes import AttributeDict


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

@dataclass
class HuggingFace:
    models = AttributeDict({
        'wav2vec2': AttributeDict({
            'model_url': 'aniemore/wav2vec2-emotion-russian-resd',
            'model_cls': Wav2Vec2ForSequenceClassification
        }),
        'wav2vec2_custom': AttributeDict({
            'model_url': 'aniemore/wav2vec2-xlsr-53-russian-emotion-recognition',
            'model_cls': AutoModelForSequenceClassification
        }),
        'wavlm': AttributeDict({
            'model_url': 'aniemore/wavlm-emotion-russian-resd',
            'model_cls': WavLMForSequenceClassification
        }),
        'hubert': AttributeDict({
            'model_url': 'aniemore/hubert-emotion-russian-resd',
            'model_cls': HubertForSequenceClassification
        }),
        'unispeech': AttributeDict({
            'model_url': 'aniemore/unispeech-emotion-russian-resd',
            'model_cls': UniSpeechSatForSequenceClassification
        }),
        'rubert_tiny': AttributeDict({
            'model_url': 'aniemore/rubert-tiny-emotion-russian-cedr-m7',
            'model_cls': BertForSequenceClassification
        }),
        'rubert_base': AttributeDict({
            'model_url': 'aniemore/rubert-base-emotion-russian-cedr-m7',
            'model_cls': BertForSequenceClassification
        }),
        'rubert_large': AttributeDict({
            'model_url': 'aniemore/rubert-large-emotion-russian-cedr-m7',
            'model_cls': BertForSequenceClassification
        }),
        'rubert_tiny2': AttributeDict({
            'model_url': 'aniemore/rubert-tiny2-russian-emotion-detection',
            'model_cls': BertForSequenceClassification
        }),
    })

    datasets = AttributeDict({
        'resd': 'aniemore/resd',
        'resd_annotated': 'aniemore/resd-annotated',
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

