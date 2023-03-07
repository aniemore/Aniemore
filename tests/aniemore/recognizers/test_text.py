"""Tests for text module
"""
import pytest

import aniemore.models
from aniemore.recognizers.text import TextRecognizer, TextEnhancer

GENERAL_TEXT_MODULE = aniemore.models.HuggingFaceModel.Text.Bert_Tiny.Bert_Tiny


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
    emotion = text_module.predict("Какой же сегодня прекрасный день, братья", single_label=True)
    assert emotion[0] == 'happiness'


def test_predict_one_sequence_emotions():
    text_module = TextRecognizer(model=GENERAL_TEXT_MODULE)
    emotions = text_module.predict("Какой же сегодня прекрасный день, братья", single_label=False)
    assert max(emotions[0], key=emotions[0].get) == 'happiness'


def test_predict_many_sequence_emotion():
    text_module = TextRecognizer(model=GENERAL_TEXT_MODULE)
    text = ['Какой же сегодня прекрасный день, братья', 'Мама, я не хочу умирать...']
    emotion = text_module.predict(text, single_label=True)
    assert emotion[0] == ['Какой же сегодня прекрасный день, братья', 'happiness'] \
           and emotion[1] == ['Мама, я не хочу умирать...', 'sadness']


def test_predict_many_sequence_emotions():
    text_module = TextRecognizer(model=GENERAL_TEXT_MODULE)
    text = ['Какой же сегодня прекрасный день, братья', 'Мама, я не хочу умирать...']
    emotions = text_module.predict(text, single_label=False)
    assert emotions[0][0] == 'Какой же сегодня прекрасный день, братья' \
           and max(emotions[0][1], key=emotions[0][1].get) == 'happiness' \
           and emotions[1][0] == 'Мама, я не хочу умирать...' \
           and max(emotions[1][1], key=emotions[1][1].get) == 'sadness'


def test_text_enhancement():
    text_module = TextEnhancer()
    text = 'какой же сегодня прекрасный день брат'
    # that's how it works, but it's not correct
    # TODO: find more reliable models
    assert text_module.enhance(text) == 'Какой же сегодня прекрасный день брат!'
