import pytest
from aniemore.recognizer.text import TextRecognizer

# TODO: do more test for
#   1. audio
#   2. torch.device control
#   3. yandex cloud speech to text?
#   4. text enhancement
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
            and max(emotions[1][1], key=emotions[1][1].get) == 'sadness' \

