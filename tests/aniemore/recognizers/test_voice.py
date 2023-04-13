"""Tests for voice module
"""
import pytest
import torch
from pathlib import Path

from aniemore.recognizers.voice import VoiceRecognizer
from aniemore.utils.speech2text import SmallSpeech2Text
from aniemore.models import HuggingFaceModel


TESTS_DIR = Path(__file__).parent
TEST_VOICE_DATA_PATH = str(TESTS_DIR / 'src' / 'my_voice.ogg')

GENERAL_WAV2VEC_MODEL = HuggingFaceModel.Voice.Wav2Vec2


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
        VoiceRecognizer()


def test_create_dummy_wav2vec2():
    vr = VoiceRecognizer(model=GENERAL_WAV2VEC_MODEL)

    assert vr.model_url == HuggingFaceModel.Voice.Wav2Vec2.model_url

    assert vr.model_cls == HuggingFaceModel.Voice.Wav2Vec2.model_cls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a runner with CUDA Compiled")
def test_device_setter():
    vr = VoiceRecognizer(model=GENERAL_WAV2VEC_MODEL)
    assert vr.device == 'cpu'
    vr.device = 'cuda'
    assert vr.device == 'cuda'
    vr.device = 'cuda:0'
    assert vr.device == 'cuda:0'
    with pytest.raises(ValueError):
        vr.device = 'cudah:0'
    with pytest.raises(ValueError):
        vr.device = 'cudah'


def test_predict_one_sequence_emotion():
    vr = VoiceRecognizer(model=GENERAL_WAV2VEC_MODEL)
    emotion = vr.recognize(TEST_VOICE_DATA_PATH)

    # check return type
    assert type(emotion) == dict


def test_predict_many_sequence_emotion():
    vr = VoiceRecognizer(model=GENERAL_WAV2VEC_MODEL)
    emotions = vr.recognize([TEST_VOICE_DATA_PATH, TEST_VOICE_DATA_PATH])

    # check return type
    assert type(emotions) == dict


def test_single_label_on_one():
    vr = VoiceRecognizer(model=GENERAL_WAV2VEC_MODEL)
    emotion = vr.recognize(TEST_VOICE_DATA_PATH, return_single_label=True)

    # check return type
    assert type(emotion) == str


def test_single_label_on_many():
    vr = VoiceRecognizer(model=GENERAL_WAV2VEC_MODEL)
    emotions = vr.recognize([TEST_VOICE_DATA_PATH, TEST_VOICE_DATA_PATH], return_single_label=True)

    # check return type
    assert type(emotions) == dict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a runner with CUDA Compiled")
def test_context_manager():
    vr = VoiceRecognizer(model=GENERAL_WAV2VEC_MODEL)

    with vr.on_device('cuda:0'):
        # check device
        assert str(vr._model.device) == 'cuda:0'
        emotion = vr.recognize(TEST_VOICE_DATA_PATH)

    # check return type
    assert type(emotion) == dict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a runner with CUDA Compiled")
def test_one_to_many_context_manager():
    vr = VoiceRecognizer(model=HuggingFaceModel.Voice.Wav2Vec2)

    with vr.on_device('cuda:0'):
        # check device
        assert str(vr._model.device) == 'cuda:0'
        emotion = vr.recognize(TEST_VOICE_DATA_PATH)

    # check return type
    assert type(emotion) == dict

    with vr.on_device('cuda:0'):
        # check device
        assert str(vr._model.device) == 'cuda:0'
        emotion = vr.recognize(TEST_VOICE_DATA_PATH)

    # check return type
    assert type(emotion) == dict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a runner with CUDA Compiled")
def test_many_to_many_context_manager():
    vr1 = VoiceRecognizer(model=GENERAL_WAV2VEC_MODEL, device='cpu')
    vr2 = VoiceRecognizer(model=GENERAL_WAV2VEC_MODEL, device='cpu')
    vr3 = VoiceRecognizer(model=GENERAL_WAV2VEC_MODEL, device='cpu')

    with vr1.on_device('cuda:0'):
        # check devices of models in handlers
        assert str(vr1._model.device) == 'cuda:0'
        assert str(vr2._model.device) == 'cpu'
        assert str(vr3._model.device) == 'cpu'

        emotion = vr1.recognize(TEST_VOICE_DATA_PATH)

    # check return type
    assert type(emotion) == dict

    with vr2.on_device('cuda:0'):
        # check devices of models in handlers
        assert str(vr1._model.device) == 'cpu'
        assert str(vr2._model.device) == 'cuda:0'
        assert str(vr3._model.device) == 'cpu'

        emotion = vr2.recognize(TEST_VOICE_DATA_PATH)

    # check return type
    assert type(emotion) == dict

    with vr3.on_device('cuda:0'):
        # check devices of models in handlers
        assert str(vr1._model.device) == 'cpu'
        assert str(vr2._model.device) == 'cpu'
        assert str(vr3._model.device) == 'cuda:0'

        emotion = vr3.recognize(TEST_VOICE_DATA_PATH)

    # check return type
    assert type(emotion) == dict


def test_load_speech_to_text():
    s2t_model = SmallSpeech2Text()
    assert s2t_model(TEST_VOICE_DATA_PATH).language == 'ru'


def test_switch_model():
    vr = VoiceRecognizer(model=HuggingFaceModel.Voice.Wav2Vec2)

    assert vr.model_url == HuggingFaceModel.Voice.Wav2Vec2.model_url

    with vr.with_model(HuggingFaceModel.Voice.WavLM, device='cpu') as new_vr:
        assert new_vr.model_url == HuggingFaceModel.Voice.WavLM.model_url
