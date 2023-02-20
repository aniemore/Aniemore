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


class Dataset(NamedTuple):
    dataset_url: str


class TextModel(Model):
    ...


class VoiceModel(Model):
    ...


class Voice(Model, Enum):
    Wav2Vec2 = VoiceModel(Wav2Vec2ForSequenceClassification, 'aniemore/wav2vec2-emotion-russian-resd')
    Wav2Vec2_Custom = VoiceModel(AutoModelForSequenceClassification,
                                 'aniemore/wav2vec2-xlsr-53-russian-emotion-recognition')
    WavLM = VoiceModel(WavLMForSequenceClassification, 'aniemore/wavlm-emotion-russian-resd')
    Hubert = VoiceModel(HubertForSequenceClassification, 'aniemore/hubert-emotion-russian-resd')
    UniSpeech = VoiceModel(UniSpeechSatForSequenceClassification, 'aniemore/unispeech-emotion-russian-resd')


class Text(Model, Enum):
    Bert_Tiny = TextModel(BertForSequenceClassification, 'aniemore/rubert-tiny-emotion-russian-cedr-m7')
    Bert_Tiny2 = TextModel(BertForSequenceClassification, 'aniemore/rubert-tiny-emotion-russian-cedr-m7-2')
    Bert_Base = TextModel(BertForSequenceClassification, 'aniemore/rubert-base-emotion-russian-cedr-m7')
    Bert_Large = TextModel(BertForSequenceClassification, 'aniemore/rubert-large-emotion-russian-cedr-m7')


class HuggingFaceModel(Model, Enum):
    Wav2Vec2 = Voice.Wav2Vec2
    Wav2Vec2_Custom = Voice.Wav2Vec2_Custom
    WavLM = Voice.WavLM
    Hubert = Voice.Hubert
    UniSpeech = Voice.UniSpeech
    Bert_Tiny = Text.Bert_Tiny
    Bert_Tiny2 = Text.Bert_Tiny2
    Bert_Base = Text.Bert_Base
    Bert_Large = Text.Bert_Large


class HuggingFaceDataset(Dataset, Enum):
    RESD = Dataset('aniemore/resd')
    RESD_ANNOTATED = Dataset('aniemore/resd-annotated')
    CEDR_M7 = Dataset('aniemore/cedr-m7')
