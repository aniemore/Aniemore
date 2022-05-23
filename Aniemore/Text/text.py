import torch
from transformers import BertForSequenceClassification, AutoTokenizer
import yaml


class EmotionFromText:
    """
    Using pre-training rubert-tiny2 model on MODIFIED CEDR dataset.
    You can see emotion list in config.yml
    """

    tokenizer = AutoTokenizer.from_pretrained('Aniemore/rubert-tiny2-russian-emotion-detection')
    model = BertForSequenceClassification.from_pretrained('Aniemore/rubert-tiny2-russian-emotion-detection')

    def __init__(self):
        try:
            with open('../config.yml', 'r') as config_file:
                self.configs = yaml.safe_load(config_file)
        except yaml.YAMLError as ex:
            print(ex)

    @torch.no_grad()
    def predict_emotion(self, text: str) -> str:
        """
            We take the input text, tokenize it, pass it through the model, and then return the predicted label

            :param text: The text to be classified
            :type text: str
            :return: The predicted emotion
        """
        inputs = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)
        predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(predicted, dim=1).numpy()

        return self.get_label_str(predicted[0])

    @torch.no_grad()
    def predict_emotions(self, text: str) -> list:
        """
            It takes a string of text, tokenizes it, feeds it to the model, and returns a dictionary of emotions and their
            probabilities

            :param text: The text you want to classify
            :type text: str
            :return: A dictionary of emotions and their probabilities.
        """
        inputs = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)
        predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
        emotions_dict = {}
        for i in range(len(predicted.numpy()[0].tolist())):
            emotions_dict[self.get_label_str(i)] = predicted.numpy()[0].tolist()[i]
        return emotions_dict

    def get_label_str(self, label_id: int) -> str:
        """
        It takes in a label id and returns the corresponding label string

        :param label_id: The label id of the label you want to get the string for
        :type label_id: int
        :return: The label string for the given label id.
        """
        return self.configs['Text']['LABELS'][label_id]
