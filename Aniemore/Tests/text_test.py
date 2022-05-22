from Aniemore.Text.text import Text
import torch

text_module = Text()
emotions = text_module.predict_emotions("Какой же сегодня прекрасный день, братья")
emotion = max(emotions)
print(emotion)