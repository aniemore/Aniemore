"""
Test for text module
"""
import pytest
from aniemore.recognizer.text import TextRecognizer, TextEnhancer


# TODO: do more test for
#   1.  audio
#   2. yandex cloud speech to text?
#   3. text enhancement

def test_device():
    # Should raise ValueError
    with pytest.raises(ValueError):
        TextRecognizer(device='cuda:')
    with pytest.raises(ValueError):
        assert TextRecognizer(device='cucu').device == 'cucu'
    with pytest.raises(ValueError):
        tr = TextRecognizer()
        tr.device = 'cucu'
        assert tr.device == 'cucu'
    # Should be fine
    assert TextRecognizer(device='cpu').device == 'cpu'
    assert TextRecognizer(device='cuda').device == 'cuda'
    assert TextRecognizer(device='cuda:0').device == 'cuda:0'


def test_predict_one_sequence_emotion():
    text_module = TextRecognizer()
    emotion = text_module.predict("Какой же сегодня прекрасный день, братья", single_label=True)
    assert emotion[0] == 'happiness'


def test_predict_one_sequence_emotions():
    text_module = TextRecognizer()
    emotions = text_module.predict("Какой же сегодня прекрасный день, братья", single_label=False)
    assert max(emotions[0], key=emotions[0].get) == 'happiness'


def test_predict_many_sequence_emotion():
    text_module = TextRecognizer()
    text = ['Какой же сегодня прекрасный день, братья', 'Мама, я не хочу умирать...']
    emotion = text_module.predict(text, single_label=True)
    assert emotion[0] == ['Какой же сегодня прекрасный день, братья', 'happiness'] \
           and emotion[1] == ['Мама, я не хочу умирать...', 'sadness']


def test_predict_many_sequence_emotions():
    text_module = TextRecognizer()
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


def test_keyword_with():
    text_module = TextRecognizer()
    assert 'text_module' in dir()
    del text_module
    assert 'text_module' not in dir()

    text_module = TextEnhancer()
    assert 'text_module' in dir()
    del text_module
    assert 'text_module' not in dir()
