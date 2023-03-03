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
    dataset_name: str
    dataset_url: str


class HuggingFaceModel:
    class Voice(Model, Enum):
        Wav2Vec2 = Model(Wav2Vec2ForSequenceClassification, 'aniemore/wav2vec2-emotion-russian-resd')
        Wav2Vec2_Custom = Model(AutoModelForSequenceClassification,
                                     'aniemore/wav2vec2-xlsr-53-russian-emotion-recognition')
        WavLM = Model(WavLMForSequenceClassification, 'aniemore/wavlm-emotion-russian-resd')
        Hubert = Model(HubertForSequenceClassification, 'aniemore/hubert-emotion-russian-resd')
        UniSpeech = Model(UniSpeechSatForSequenceClassification, 'aniemore/unispeech-emotion-russian-resd')

    class Text(Model, Enum):
        Bert_Tiny = Model(BertForSequenceClassification, 'aniemore/rubert-tiny-emotion-russian-cedr-m7')
        Bert_Tiny2 = Model(BertForSequenceClassification, 'aniemore/rubert-tiny-emotion-russian-cedr-m7-2')
        Bert_Base = Model(BertForSequenceClassification, 'aniemore/rubert-base-emotion-russian-cedr-m7')
        Bert_Large = Model(BertForSequenceClassification, 'aniemore/rubert-large-emotion-russian-cedr-m7')


class HuggingFaceDataset(Dataset, Enum):
    RESD = Dataset('Russian Emotional Speech Dialoges', 'aniemore/resd')
    RESD_ANNOTATED = Dataset('Russian Emotional Speech Dialoges [Annotated]', 'aniemore/resd-annotated')
    CEDR_M7 = Dataset('Corpus for Emotions Detecting moods 7', 'aniemore/cedr-m7')
