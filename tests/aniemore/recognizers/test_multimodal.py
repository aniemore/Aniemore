"""Tests for voice module
"""
from typing import List

import pytest
import torch
from pathlib import Path

from aniemore.recognizers.multimodal import VoiceTextRecognizer, MultiModalRecognizer
from aniemore.utils.speech2text import SmallSpeech2Text, Speech2Text
from aniemore.models import HuggingFaceModel

TESTS_DIR = Path(__file__).parent
TEST_VOICE_DATA_PATH = str(TESTS_DIR / 'src' / 'my_voice.ogg')

GENERAL_WAVLM_BERT_MODEL = HuggingFaceModel.MultiModal.WavLMBertFusion


@pytest.fixture(autouse=True)
def run_around_test():
    # would be run before test
    yield  # exact test happens
    # would be run after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


def test_create_empty():
    with pytest.raises(AttributeError):
        MultiModalRecognizer()


def test_create_dummy_voice_text():
    vtr = VoiceTextRecognizer(model=GENERAL_WAVLM_BERT_MODEL)

    assert vtr.model_url == GENERAL_WAVLM_BERT_MODEL.model_url

    assert vtr.model_cls == GENERAL_WAVLM_BERT_MODEL.model_cls

    del vtr


def test_create_dummy_multimodal():
    mr = MultiModalRecognizer(model=GENERAL_WAVLM_BERT_MODEL, s2t_model=SmallSpeech2Text())

    assert mr.model_url == GENERAL_WAVLM_BERT_MODEL.model_url

    assert mr.model_cls == GENERAL_WAVLM_BERT_MODEL.model_cls

    assert isinstance(mr.s2t_model, Speech2Text)

    del mr


def test_predict_one_sequence_emotion():
    mr = MultiModalRecognizer(model=GENERAL_WAVLM_BERT_MODEL, s2t_model=SmallSpeech2Text())
    emotion = mr.recognize(TEST_VOICE_DATA_PATH)

    # check return type
    assert type(emotion) == dict

    del mr


def test_predict_many_sequence_emotion():
    mr = MultiModalRecognizer(model=GENERAL_WAVLM_BERT_MODEL, s2t_model=SmallSpeech2Text())
    emotions = mr.recognize([TEST_VOICE_DATA_PATH, TEST_VOICE_DATA_PATH])

    # check return type
    assert type(emotions) == dict

    del mr


def test_single_label_on_one():
    mr = MultiModalRecognizer(model=GENERAL_WAVLM_BERT_MODEL, s2t_model=SmallSpeech2Text())
    emotion = mr.recognize(TEST_VOICE_DATA_PATH, return_single_label=True)

    # check return type
    assert type(emotion) == str

    del mr


def test_single_label_on_many():
    mr = MultiModalRecognizer(model=GENERAL_WAVLM_BERT_MODEL, s2t_model=SmallSpeech2Text())
    emotions = mr.recognize([TEST_VOICE_DATA_PATH, TEST_VOICE_DATA_PATH], return_single_label=True)

    # check return type
    assert type(emotions) == dict

    del mr


def test_single_top_n():
    mr = MultiModalRecognizer(model=GENERAL_WAVLM_BERT_MODEL, s2t_model=SmallSpeech2Text())
    emotions = mr.recognize(TEST_VOICE_DATA_PATH, top_n=2)

    # check return type
    assert type(emotions) == list

    del mr


def test_many_top_n():
    mr = MultiModalRecognizer(model=GENERAL_WAVLM_BERT_MODEL, s2t_model=SmallSpeech2Text())
    emotions = mr.recognize([TEST_VOICE_DATA_PATH, TEST_VOICE_DATA_PATH], top_n=2)

    # check return type
    assert type(emotions) == dict

    del mr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a runner with CUDA Compiled")
def test_context_manager():
    mr = MultiModalRecognizer(model=GENERAL_WAVLM_BERT_MODEL, s2t_model=SmallSpeech2Text())

    with mr.on_device('cuda:0'):
        # check device
        assert str(mr._model.device) == 'cuda:0'
        emotion = mr.recognize(TEST_VOICE_DATA_PATH)

    # check return type
    assert type(emotion) == dict

    del mr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a runner with CUDA Compiled")
def test_one_to_many_context_manager():
    mr = MultiModalRecognizer(model=GENERAL_WAVLM_BERT_MODEL, s2t_model=SmallSpeech2Text())

    with mr.on_device('cuda:0'):
        # check device
        assert str(mr._model.device) == 'cuda:0'
        emotion = mr.recognize(TEST_VOICE_DATA_PATH)

    # check return type
    assert type(emotion) == dict

    with mr.on_device('cuda:0'):
        # check device
        assert str(mr._model.device) == 'cuda:0'
        emotion = mr.recognize(TEST_VOICE_DATA_PATH)

    # check return type
    assert type(emotion) == dict

    del mr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a runner with CUDA Compiled")
def test_many_to_many_context_manager():
    mr1 = MultiModalRecognizer(model=GENERAL_WAVLM_BERT_MODEL, s2t_model=SmallSpeech2Text())
    mr2 = MultiModalRecognizer(model=GENERAL_WAVLM_BERT_MODEL, s2t_model=SmallSpeech2Text())

    with mr1.on_device('cuda:0'):
        # check devices of models in handlers
        assert str(mr1._model.device) == 'cuda:0'
        assert str(mr2._model.device) == 'cpu'

        emotion = mr1.recognize(TEST_VOICE_DATA_PATH)

    # check return type
    assert type(emotion) == dict

    with mr2.on_device('cuda:0'):
        # check devices of models in handlers
        assert str(mr1._model.device) == 'cpu'
        assert str(mr2._model.device) == 'cuda:0'

        emotion = mr2.recognize(TEST_VOICE_DATA_PATH)

    # check return type
    assert type(emotion) == dict

    del mr1, mr2
