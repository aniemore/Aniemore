import pytest
from aniemore.recognizer.voice import VoiceRecognizer
from aniemore.config_enums import HuggingFaceModel
from aniemore.utils.custom_classes import ModelOutput, RecognizerOutputRepr


def test_create_empty():
    with pytest.raises(TypeError):
        VoiceRecognizer()


def test_create_dummy_wav2vec2():
    vr = VoiceRecognizer(model_name=HuggingFaceModel.Wav2Vec2)
    assert vr.MODEL_URL == HuggingFaceModel.Wav2Vec2.model_url
    assert vr.MODEL_CLS == HuggingFaceModel.Wav2Vec2.model_cls


def test_device_setter():
    vr = VoiceRecognizer(model_name=HuggingFaceModel.Wav2Vec2)
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
    vr = VoiceRecognizer(model_name=HuggingFaceModel.Wav2Vec2)
    emotion = vr.predict("tests/aniemore/my_voice.ogg")

    # check return type
    assert type(emotion) == dict


def test_predict_many_sequence_emotion():
    vr = VoiceRecognizer(model_name=HuggingFaceModel.Wav2Vec2)
    emotions = vr.predict(["tests/aniemore/my_voice.ogg", "tests/aniemore/my_voice.ogg"])

    # check return type
    assert type(emotions) == dict


def test_context_manager():
    vr = VoiceRecognizer(model_name=HuggingFaceModel.Wav2Vec2)

    with vr.on_device('cuda:0'):
        emotion = vr.predict("tests/aniemore/my_voice.ogg")

    # check return type
    assert type(emotion) == dict
