import json
import yaml
import torch
from os import popen
from time import time
import requests as requests

from aniemore import config


class SpeechToText:
    """
    SpeechToText класс.
    Обязательно убедитесь, что у вас установлен и настроен YandexCloud-CLI.
    """

    grammar_model, apply_te = {}, {}

    def __init__(self, yandex_cloud_folder_id: str):
        """
        Конструктор класса. Обязательно убедитесь, что у вас установлен и настроен YandexCloud-CLI.
        Берём предоставленный YandexCloud FolderID и сохраняем его в config.yml

        :param yandex_cloud_folder_id: YandexCloud FolderID
        :type yandex_cloud_folder_id: str
        """

        # TODO:
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
        # TODO: мб просто функцию сделать для настройки под мак и старой версии торча :\
        # Только для Apple M1 для версий torch, не использующих GPU. Если под другими процессорами - раскомментировать.
        torch.backends.quantized.engine = 'qnnpack'
        self.grammar_model, _, _, _, self.apply_te = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                    model='silero_te')

    def recognize(self, sound_data: bytes) -> dict:
        """
        Мы отправляем POST запрос на сервер Yandex Cloud, содержащий аудио файл, и получаем JSON строку с текстом

        :param sound_data: binary audio (.ogg файл, короче 30s, только 48kHz и 16kbps)
        :type sound_data: bytes
        :return: текст из аудио файла
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
        Получаем строку, добавляем грамматику и возвращаем строку по правилам русского языка

        :param text: текст, который нужно подкорректировать
        :type text: str
        :return: подкорректированный текст
        """
        text = text.get("result", "").lower()
        grammar_text = self.apply_te(text, lan="ru")
        return grammar_text

    def __get_iam_token(self):
        """
        Если токена нет в config.yml, или он старше 12 часов -> создаем новый и сохраняем в config.yml,
        а возвращаем валидный iam токен.
        :return: YandexCloud IAM-token
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

