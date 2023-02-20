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


def test_one_to_many_context_manager():
    vr = VoiceRecognizer(model_name=HuggingFaceModel.Wav2Vec2)

    with vr.on_device('cuda:0'):
        emotion = vr.predict("tests/aniemore/my_voice.ogg")

    # check return type
    assert type(emotion) == dict

    with vr.on_device('cuda:0'):
        emotion = vr.predict("tests/aniemore/my_voice.ogg")

    # check return type
    assert type(emotion) == dict


def test_many_to_many_context_manager():
    vr1 = VoiceRecognizer(model_name=HuggingFaceModel.Wav2Vec2, device='cuda:0')
    vr2 = VoiceRecognizer(model_name=HuggingFaceModel.Wav2Vec2, device='cuda:0')
    vr3 = VoiceRecognizer(model_name=HuggingFaceModel.Wav2Vec2, device='cuda:0')

    with vr1.on_device('cuda:0'):
        # check devices of models in handlers
        assert vr1.model.device == 'cuda:0'
        assert vr2.model.device == 'cpu'
        assert vr3.model.device == 'cpu'

        emotion = vr1.predict("tests/aniemore/my_voice.ogg")

    # check return type
    assert type(emotion) == dict

    with vr2.on_device('cuda:0'):
        # check devices of models in handlers
        assert vr1.model.device == 'cpu'
        assert vr2.model.device == 'cuda:0'
        assert vr3.model.device == 'cpu'

        emotion = vr2.predict("tests/aniemore/my_voice.ogg")

    # check return type
    assert type(emotion) == dict

    with vr3.on_device('cuda:0'):
        # check devices of models in handlers
        assert vr1.model.device == 'cpu'
        assert vr2.model.device == 'cpu'
        assert vr3.model.device == 'cuda:0'

        emotion = vr3.predict("tests/aniemore/my_voice.ogg")

    # check return type
    assert type(emotion) == dict
