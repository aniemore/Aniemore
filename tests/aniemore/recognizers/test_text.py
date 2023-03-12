"""Tests for text module
"""
import pytest
import torch

import aniemore.models
from aniemore.recognizers.text import TextRecognizer, TextEnhancer

GENERAL_TEXT_MODULE = aniemore.models.HuggingFaceModel.Text.Bert_Tiny.Bert_Tiny


@pytest.fixture(autouse=True)
def run_around_test():
    # would be run before test
    yield  # exact test happens
    # would be run after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a runner with CUDA Compiled")
def test_device():
    # Should raise ValueError
    with pytest.raises(ValueError):
        TextRecognizer(model=GENERAL_TEXT_MODULE, device='cuda:')
    with pytest.raises(ValueError):
        assert TextRecognizer(model=GENERAL_TEXT_MODULE, device='cucu').device == 'cucu'
    with pytest.raises(ValueError):
        tr = TextRecognizer(model=GENERAL_TEXT_MODULE)
        tr.device = 'cucu'
        assert tr.device == 'cucu'
    # Should be fine
    assert TextRecognizer(model=GENERAL_TEXT_MODULE, device='cpu').device == 'cpu'
    assert TextRecognizer(model=GENERAL_TEXT_MODULE, device='cuda').device == 'cuda'
    assert TextRecognizer(model=GENERAL_TEXT_MODULE, device='cuda:0').device == 'cuda:0'


def test_predict_one_sequence_emotion():
    text_module = TextRecognizer(model=GENERAL_TEXT_MODULE)
    emotion = text_module.recognize("Какой же сегодня прекрасный день, братья", return_single_label=True)
    assert emotion == 'happiness'


def test_predict_one_sequence_emotions():
    text_module = TextRecognizer(model=GENERAL_TEXT_MODULE)
    emotions = text_module.recognize("Какой же сегодня прекрасный день, братья", return_single_label=False)
    assert max(emotions, key=emotions.get) == 'happiness'


def test_predict_many_sequence_emotion():
    text_module = TextRecognizer(model=GENERAL_TEXT_MODULE)
    text = ['Какой же сегодня прекрасный день, братья', 'Мама, я не хочу умирать...']
    o_emotions = ['happiness', 'sadness']
    emotions = text_module.recognize(text, return_single_label=True)
    for original, (text_, emotion) in zip(text, emotions.items()):
        assert original == text_ and emotion == o_emotions.pop(0)


def test_predict_many_sequence_emotions():
    text_module = TextRecognizer(model=GENERAL_TEXT_MODULE)
    text = ['Какой же сегодня прекрасный день, братья', 'Мама, я не хочу умирать...']
    o_emotions = ['happiness', 'sadness']
    emotions = text_module.recognize(text, return_single_label=False)
    for original, (text_, emotion) in zip(text, emotions.items()):
        assert original == text_ and max(emotion, key=emotion.get) == o_emotions.pop(0)


def test_text_enhancement():
    text_module = TextEnhancer()
    text = 'какой же сегодня прекрасный день брат'
    # that's how it works, but it's not correct
    # TODO: find more reliable models
    assert text_module.enhance(text) == 'Какой же сегодня прекрасный день брат!'
