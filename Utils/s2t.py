import json
from os import popen
import requests as requests
import settings

class SpeechToText:
    """
    Speech to text main class
    """
    def recognize(self, sound_data, ID_FOLDER):
        """
        :param sound_data: .ogg file, less than 30s, only 48kHz, only 16 kbps
        :param ID_FOLDER: your yandex cloud folder id
        :return: string of recognized text
        """
        iam_token = popen("yc iam create-token").read().replace("\n", "")

        # в поле заголовка передаем IAM_TOKEN:
        headers = {'Authorization': f'Bearer {iam_token}'}

        # остальные параметры:
        params = {
            'lang': 'ru-RU',
            'folderId': ID_FOLDER,
            'sampleRateHertz': 48000,
        }

        # Делаем запрос на сервера SpeechKit
        response = requests.post(settings.YA_URL, params=params, headers=headers, data=sound_data)

        # бинарные ответ доступен через response.content, декодируем его:
        decode_resp = response.content.decode('UTF-8')

        # и загрузим в json, чтобы получить текст из аудио:
        text = json.loads(decode_resp)

        return text

# TODO: Remove all below later (test segment)
s2t = SpeechToText()
text = s2t.recognize(open('/Users/toiletsandpaper/Downloads/my_voice(2).ogg', 'rb').read(), 'b1g75n4nfd17pefq9el4')
print(text)
