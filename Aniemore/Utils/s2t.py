import json
import yaml
import torch
from os import popen
from time import time
import requests as requests


class SpeechToText:
    """
    Speech to text main class
    """

    configs = {}
    grammar_model, apply_te = {}, {}

    def __init__(self, yandex_cloud_folder_id: str):
        """
        Init method for SpeechToText.
        It takes a Yandex Cloud folder ID as an argument and saves it to the config file

        :param yandex_cloud_folder_id: the ID of your folder in Yandex.Cloud
        :type yandex_cloud_folder_id: str
        """

        try:
            with open('../config.yml', 'r') as config_file:
                self.configs = yaml.safe_load(config_file)
        except yaml.YAMLError as ex:
            print(ex)

        self.configs['YandexCloud']['YC_FOLDER_ID'] = yandex_cloud_folder_id
        try:
            with open('../config.yml', 'w') as config_file:
                yaml.safe_dump(self.configs, config_file)
        except yaml.YAMLError as ex:
            print(ex)
        # Only Apple M1 fixes, comment if under other OS
        # torch.backends.quantized.engine = 'qnnpack'
        self.grammar_model, _, _, _, self.apply_te = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                    model='silero_te')

    def recognize(self, sound_data: bytes) -> dict:
        """
        We send a POST request to the Yandex Cloud server with the audio file, and get a JSON response with the text

        :param sound_data: binary audio (.ogg file, less than 30s, only 48kHz, only 16 kbps)
        :type sound_data: bytes
        :return: The text of the audio file.
        """
        headers = {'Authorization': f'Bearer {self.__get_iam_token()}'}

        params = {
            'lang': 'ru-RU',
            'folderId': self.configs['YandexCloud']['YC_FOLDER_ID'],
            'sampleRateHertz': 48000,
        }

        response = requests.post(self.configs['YandexCloud']['YC_URL'], params=params, headers=headers, data=sound_data)
        decode_resp = response.content.decode('UTF-8')
        text = json.loads(decode_resp)

        if text.get('result') is None:
            raise Exception(text)

        return text

    def echance_text(self, text: str) -> str:
        """
        It takes a string, applies a grammar to it, and returns the result

        :param text: The text to be corrected
        :type text: str
        :return: The text is being returned.
        """
        text = text.get("result", "").lower()
        grammar_text = self.apply_te(text, lan="ru")
        return grammar_text

    def __get_iam_token(self):
        """
        If the token is not set or it's older than 12 hours, create a new one and save it to the config file
        :return: The token is being returned.
        """
        if (
                self.configs['YandexCloud']['YC_IAM_TOKEN']['token'] is None or
                self.configs['YandexCloud']['YC_IAM_TOKEN']['token'] == '' or
                self.configs['YandexCloud']['YC_IAM_TOKEN']['created_at'] is None or
                abs(self.configs['YandexCloud']['YC_IAM_TOKEN']['created_at'] - time()) >= 3600 * 12
        ):
            # Make shure, that YandexCloud-cli is installed, added to PATH and configured properly
            self.configs['YandexCloud']['YC_IAM_TOKEN']['token'] = popen("yc iam create-token").read().replace("\n", "")
            self.configs['YandexCloud']['YC_IAM_TOKEN']['created_at'] = time()
            try:
                with open('../config.yml', 'w') as config_file:
                    yaml.safe_dump(self.configs, config_file)
            except yaml.YAMLError as ex:
                print(ex)

        return self.configs['YandexCloud']['YC_IAM_TOKEN']['token']
