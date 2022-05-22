import torch
from transformers import BertForSequenceClassification, AutoTokenizer


class Text:
    LABELS = ['neutral', 'happiness', 'sadness', 'enthusiasm', 'fear', 'anger', 'disgust']
    tokenizer = AutoTokenizer.from_pretrained('rubert-tiny2-russian-emotion-detection')
    model = BertForSequenceClassification.from_pretrained('rubert-tiny2-russian-emotion-detection')

    def __init__(self):
        ...

    @torch.no_grad()
    def predict_emotion(self, text) -> str:
        inputs = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)
        predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(predicted, dim=1).numpy()

        return self.LABELS[predicted[0]]

    @torch.no_grad()
    def predict_emotions(self, text) -> list:
        inputs = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)
        predicted = torch.nn.functional.softmax(outputs.logits, dim=1)

        return predicted.numpy()[0].tolist()
