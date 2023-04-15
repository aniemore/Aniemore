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

from aniemore.custom.models import (
    Wav2Vec2BertForSequenceClassification,
    WavLMBertForSequenceClassification
)


class Model(NamedTuple):
    """
    NamedTuple класс, используемый для более удобного доступа к информации о модели

    Attributes:
     model_cls: HuggingFace класс модели
     model_url: ссылка на модель на HuggingFace

    Examples:
     >>> from aniemore.models import Model
     >>> my_model: Model = Model(BertForSequenceClassification, 'user/model-repo-link')
     >>> # load `my_model` to recognizer class
    """
    model_cls: Type[PreTrainedModel]
    model_url: str


@dataclasses.dataclass(frozen=True)
class HuggingFaceModel:
    """
    References:
        Our models and datasets placed here: https://huggingface.co/Aniemore
    """
    class Voice(Model, Enum):
        """
            Attributes:
                 Wav2Vec2: '`niemore/wav2vec2-emotion-russian-resd`
                 Wav2Vec2_Custom: `aniemore/wav2vec2-xlsr-53-russian-emotion-recognition`
                 WavLM: `aniemore/wavlm-emotion-russian-resd`
                 Hubert: `aniemore/hubert-emotion-russian-resd`
                 UniSpeech: `aniemore/unispeech-sat-emotion-russian-resd`
            References:
                Our models and datasets placed here: https://huggingface.co/Aniemore
        """
        Wav2Vec2 = Model(Wav2Vec2ForSequenceClassification, 'aniemore/wav2vec2-emotion-russian-resd')
        Wav2Vec2_Custom = Model(
            AutoModelForSequenceClassification,
            'aniemore/wav2vec2-xlsr-53-russian-emotion-recognition'
        )
        WavLM = Model(WavLMForSequenceClassification, 'aniemore/wavlm-emotion-russian-resd')
        Hubert = Model(HubertForSequenceClassification, 'aniemore/hubert-emotion-russian-resd')
        UniSpeech = Model(UniSpeechSatForSequenceClassification, 'aniemore/unispeech-sat-emotion-russian-resd')

    class Text(Model, Enum):
        """
            Attributes:
                Bert_Tiny: `aniemore/rubert-tiny2-russian-emotion-detection`
                Bert_Tiny2: `aniemore/rubert-tiny-emotion-russian-cedr-m7`
                Bert_Base: `aniemore/rubert-base-emotion-russian-cedr-m7`
                Bert_Large: `aniemore/rubert-large-emotion-russian-cedr-m7`
            References:
                Our models and datasets placed here: https://huggingface.co/Aniemore
        """
        Bert_Tiny2 = Model(BertForSequenceClassification, 'aniemore/rubert-tiny2-russian-emotion-detection')
        Bert_Tiny = Model(BertForSequenceClassification, 'aniemore/rubert-tiny-emotion-russian-cedr-m7')
        Bert_Base = Model(BertForSequenceClassification, 'aniemore/rubert-base-emotion-russian-cedr-m7')
        Bert_Large = Model(BertForSequenceClassification, 'aniemore/rubert-large-emotion-russian-cedr-m7')

    class MultiModal(Model, Enum):
        """
            Attributes:
                Wav2Vec2BertTiny: `aniemore/wav2vec2-bert-tiny2-emotion-russian-resd`
                Wav2Vec2BertBase: `aniemore/wav2vec2-bert-base-emotion-russian-resd`
                WavLMBertTiny: `aniemore/wavlm-bert-tiny2-emotion-russian-resd`
                WavLMBertBase: `Ar4ikov/wavlm-bert-base-multimodal-emotion-russian-resd`
            References:
                Our models and datasets placed here: https://huggingface.co/Aniemore
        """
        Wav2Vec2BertTiny = Model(
            Wav2Vec2BertForSequenceClassification, 'aniemore/wav2vec2-bert-tiny2-s-emotion-russian-resd')
        Wav2Vec2BertBase = Model(
            Wav2Vec2BertForSequenceClassification, 'aniemore/wav2vec2-bert-base-s-emotion-russian-resd')
        WavLMBertTiny = Model(
            WavLMBertForSequenceClassification, 'aniemore/wavlm-bert-tiny2-s-emotion-russian-resd')
        WavLMBertBase = Model(
            WavLMBertForSequenceClassification, 'aniemore/wavlm-bert-base-s-emotion-russian-resd')
