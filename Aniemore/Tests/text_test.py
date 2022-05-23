from Aniemore.Text.text import EmotionFromText
import torch

text_module = EmotionFromText()
emotions = text_module.predict_emotions("Какой же сегодня прекрасный день, братья")
emotion = max(emotions)
print(emotion)