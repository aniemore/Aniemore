import json
import yaml
import torch
from re import sub, findall
from os import popen
from time import time
import requests as requests


class SpeechToText:
    """
    Speech to text main class
    """

    configs = {}
    grammar_model, apply_te = {}, {}

    def __init__(self):
        """
        Init method for SpeechToText. Reads configs.yml on creation.
        """
        try:
            with open('../config.yml', 'r') as config_file:
                self.configs = yaml.safe_load(config_file)
        except yaml.YAMLError as ex:
            print(ex)
        torch.backends.quantized.engine = 'qnnpack'
        self.grammar_model, _, _, _, self.apply_te = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_te')

    def recognize(self, sound_data):
        """
        :param sound_data: .ogg file, less than 30s, only 48kHz, only 16 kbps
        :param ID_FOLDER: your yandex cloud folder id
        :return: string of recognized text
        """
        self.__get_iam_token()
        # в поле заголовка передаем YC_IAM_TOKEN:
        headers = {'Authorization': f'Bearer {self.__get_iam_token()}'}

        # остальные параметры:
        params = {
            'lang': 'ru-RU',
            'folderId': self.configs['YandexCloud']['YC_FOLDER_ID'],
            'sampleRateHertz': 48000,
        }

        # Делаем запрос на сервера SpeechKit
        response = requests.post(self.configs['YandexCloud']['YC_URL'], params=params, headers=headers, data=sound_data)

        # бинарные ответ доступен через response.content, декодируем его:
        decode_resp = response.content.decode('UTF-8')

        # и загрузим в json, чтобы получить текст из аудио:
        text = json.loads(decode_resp)

        return text

    def echance_text(self, text):
        text = text.get("result", "").lower()
        grammar_text = self.apply_te(text, lan="ru")
        return grammar_text



    def __get_iam_token(self):
        """
        Gets YandexCloud IAM token (lifetime - 12 hours)
        :return: YandexCloud IAM token from config.yml
        """
        if (
            self.configs['YandexCloud']['YC_IAM_TOKEN']['token'] is None or
            self.configs['YandexCloud']['YC_IAM_TOKEN']['created_at'] is None or
            abs((self.configs['YandexCloud']['YC_IAM_TOKEN']['created_at'] - time()) >= 3600 * 12)
        ):
            self.configs['YandexCloud']['YC_IAM_TOKEN']['token'] = popen("yc iam create-token").read().replace("\n", "")
            self.configs['YandexCloud']['YC_IAM_TOKEN']['created_at'] = time()
            try:
                with open('../config.yml', 'w') as config_file:
                    yaml.safe_dump(self.configs, config_file)
            except yaml.YAMLError as ex:
                ... # print(ex)

        return self.configs['YandexCloud']['YC_IAM_TOKEN']['token']