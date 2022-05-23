from Aniemore.Text.text import EmotionFromText

text_module = EmotionFromText()
emotions = text_module.predict_emotions("Какой же сегодня прекрасный день, братья")
emotion = text_module.predict_emotion("Какой же сегодня прекрасный день, братья")
print(emotions)
print(emotion)