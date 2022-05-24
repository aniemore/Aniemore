import torch
from transformers import BertForSequenceClassification, AutoTokenizer
import yaml
from Aniemore.config import config


class EmotionFromText:
    """
    Используем уже обученную (на модифированном CEDR датасете) rubert-tiny2 модель.
    Список эмоций и их ID в модели можете посмотроеть в config.yml
    """
    MODEL_URL = config["Huggingface"]["models"]["rubert_tiny2_text"]

    tokenizer: AutoTokenizer = None
    model: BertForSequenceClassification = None

    def __init__(self):
        pass

    def setup_variables(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_URL)
        self.model = BertForSequenceClassification.from_pretrained(self.MODEL_URL)

    @torch.no_grad()
    def predict_emotion(self, text: str) -> str:
        """
            Получаем строку текста, токенизируем, отправляем в модель и возвращаем эмоцию

            :param text: текст для анализа
            :type text: str
            :return: наиболее вероятная эмоция
        """
        if self.model is None:
            self.setup_variables()

        inputs = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)
        predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(predicted, dim=1).numpy()

        return self.get_label_str(predicted[0])

    @torch.no_grad()
    def predict_emotions(self, text: str) -> dict:
        """
        Получаем строку текста, токенизируем, отправляем в модель и возвращаем лист "эмоция : вероятность"

        :param text: текст для анализа
        :type text: str
        :return: список "эмоция : вероятность"
        """
        if self.model is None:
            self.setup_variables()

        inputs = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)
        predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
        emotions_dict = {}
        for i in range(len(predicted.numpy()[0].tolist())):
            emotions_dict[self.get_label_str(i)] = predicted.numpy()[0].tolist()[i]
        return emotions_dict

    @staticmethod
    def get_label_str(label_id: int) -> str:
        """
        Берём цифру, которая выдала модель (label_id) и возвращаем соотвествующую этому ID эмоцию (строкой)

        :param label_id: label_id который выдала модель
        :type label_id: int
        :return: строка с эмоцией соотвествующую label_id
        """
        return config['Text']['LABELS'][label_id]


