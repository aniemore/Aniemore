from Aniemore.Utils.s2t import SpeechToText
import os


s2t_model = SpeechToText(yandex_cloud_folder_id='b1g75n4nfd17pefq9el4')
#s2t_model.set_yandex_cloud_folder_id()
recognized_text = s2t_model.recognize(sound_data=open(f'{os.path.dirname(__file__)}/files/my_voice.ogg', 'rb').read())
print(recognized_text['result'])
echanced_text = s2t_model.echance_text(recognized_text)
print(echanced_text)

