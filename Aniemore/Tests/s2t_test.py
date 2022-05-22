from Aniemore.Utils.s2t import SpeechToText
import os


s2t_model = SpeechToText()
recognized_text = s2t_model.recognize(sound_data=open(f'{os.path.dirname(__file__)}/files/my_voice.ogg', 'rb').read())
print(recognized_text['result'])
echanced_text = s2t_model.echance_text(recognized_text)
print(echanced_text)

