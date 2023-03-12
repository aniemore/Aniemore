"""Здесь хранятся все модели, используемые в проекте
"""
import dataclasses
from enum import Enum
from typing import NamedTuple, Type
from transformers import (
    Wav2Vec2ForSequenceClassification,
    WavLMForSequenceClassification,
    UniSpeechSatForSequenceClassification,
    HubertForSequenceClassification,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    PreTrainedModel
)


class Model(NamedTuple):
    model_cls: Type[PreTrainedModel]
    model_url: str


@dataclasses.dataclass(frozen=True)
class HuggingFaceModel:
    class Voice(Model, Enum):
        Wav2Vec2 = Model(Wav2Vec2ForSequenceClassification, 'aniemore/wav2vec2-emotion-russian-resd')
        Wav2Vec2_Custom = Model(
            AutoModelForSequenceClassification,
            'aniemore/wav2vec2-xlsr-53-russian-emotion-recognition'
        )
        WavLM = Model(WavLMForSequenceClassification, 'aniemore/wavlm-emotion-russian-resd')
        Hubert = Model(HubertForSequenceClassification, 'aniemore/hubert-emotion-russian-resd')
        UniSpeech = Model(UniSpeechSatForSequenceClassification, 'aniemore/unispeech-emotion-russian-resd')

    class Text(Model, Enum):
        Bert_Tiny = Model(BertForSequenceClassification, 'aniemore/rubert-tiny2-russian-emotion-detection')
        Bert_Tiny2 = Model(BertForSequenceClassification, 'aniemore/rubert-tiny-emotion-russian-cedr-m7')
        Bert_Base = Model(BertForSequenceClassification, 'aniemore/rubert-base-emotion-russian-cedr-m7')
        Bert_Large = Model(BertForSequenceClassification, 'aniemore/rubert-large-emotion-russian-cedr-m7')
