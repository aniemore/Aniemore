import pytest
from transformers import (
    BertForSequenceClassification,
    AutoModelForSequenceClassification,
    Wav2Vec2ForSequenceClassification,
)

from aniemore.models import Model, HuggingFaceModel


def test_model_instantiation():
    _cls_list = [
        BertForSequenceClassification,
        AutoModelForSequenceClassification,
        Wav2Vec2ForSequenceClassification,
    ]
    _url = 'dummy/url'

    for _cls in _cls_list:
        model = Model(_cls, _url)
        assert model.model_cls == _cls
        assert model.model_url == _url


def test_hg_model_accesibility_in_code():
    tmodel = HuggingFaceModel.Text.Bert_Tiny2
    assert tmodel.model_cls == BertForSequenceClassification
    assert tmodel.model_url == 'aniemore/rubert-tiny2-russian-emotion-detection'

    vmodel = HuggingFaceModel.Voice.Wav2Vec2
    assert vmodel.model_cls == Wav2Vec2ForSequenceClassification
    assert vmodel.model_url == 'aniemore/wav2vec2-emotion-russian-resd'
