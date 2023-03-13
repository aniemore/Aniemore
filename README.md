![Aniemore Logo](logo.png)

 **Aniemore** - это открытая библиотека искусственного интеллекта для потоковой аналитики эмоциональных оттенков речи человека.

#### Основные технические параметры

- Объем набора данных Russian Emotional Speech Dialogs содержит 2000 аудиофрагментов представляющих 200 различных людей;
- Модели способны распознавать эмоции в зашумленных аудиофайлах длительностью в 3 секунды;
- Скорость обработки и ответа модели составляет не более 5 секунд;
- Пословная ошибка модели WER 30%;
- Совокупная точность модели 75%
- Диапазон распознавания эмоций: злость, отвращение, страх, счастье, интерес, грусть, нейтрально;
- Акустические возможности - 3 уровня.




## Описание
Aniemore - это библиотека для Python, которая позволяет добавить в ваше 
программное обеспечение возможность определять эмоциональный фон речи человека, как в голосе, 
так и в тексте. Для этого в библиотеке разработано два соответсвующих модуля - Voice и Text.

Aniemore содержит свой собственный датасет RESD (Russian Emotional Speech Dialoges) и другие 
наборы данных разного объема, которые вы можете использовать для обучения своих моделей.

| Датасет        | Примечание                                                                  |
|----------------|-----------------------------------------------------------------------------|
| RESD           | 7 эмоций, 4 часа аудиозаписей диалогов **студийное качество**               |
| RESD_Annotated | RESD + speech-to-text аннотации                                             |
| REPV           | 2000 голосовых сообщений (.ogg), 200 актеров, 2 нейтральные фразы, 5 эмоций |
| REPV-S         | 140 голосовых сообщений (.ogg) "Привет, как дела?" с разными эмоциями       |

Вы можете использовать готовые предобученные модели из библиотеки: 

| Модель                                                                                                                           | Точность |
|----------------------------------------------------------------------------------------------------------------------------------|----------|
| Голосовые модели                                                                                                                 |          |
| [**wav2vec2-xlsr-53-russian-emotion-recognition**](https://huggingface.co/Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition) | 73%      |
| [**wav2vec2-emotion-russian-resd**](https://huggingface.co/Aniemore/wav2vec2-emotion-russian-resd)                               | 75%      |
| [**wavlm-emotion-russian-resd**](https://huggingface.co/Aniemore/wavlm-emotion-russian-resd)                                     | 82%      |
| [**unispeech-sat-emotion-russian-resd Copied**](https://huggingface.co/Aniemore/unispeech-sat-emotion-russian-resd)              | 72%      |
| Текстовые модели                                                                                                                 |          |
| [**rubert-base-emotion-russian-cedr-m7**](https://huggingface.co/Aniemore/rubert-base-emotion-russian-cedr-m7)                   | ---%     |
| [**hubert-emotion-russian-resd**](https://huggingface.co/Aniemore/hubert-emotion-russian-resd)                                   | 75%      |
| [**rubert-tiny2-russian-emotion-detection**](https://huggingface.co/Aniemore/rubert-tiny2-russian-emotion-detection)             | 85%      |
| [**rubert-large-emotion-russian-cedr-m7**](https://huggingface.co/Aniemore/rubert-large-emotion-russian-cedr-m7)                 | ---%     |
| [**rubert-tiny-emotion-russian-cedr-m7 **](https://huggingface.co/Aniemore/rubert-tiny-emotion-russian-cedr-m7)                  | ---%     |

#### Показатели моделей в разрезе эмоций
![показатели моделей.jpg](model_sota.jpg)


## <a name="Install"></a>	Установка
```shell
pip install aniemore
```
<hr>

### Пример использования
#### Распознавание эмоций в тексте
```shell
# @title Text: Bert_Tiny2
import torch
from aniemore.recognizers.text import TextRecognizer
from aniemore.models import HuggingFaceModel

model=HuggingFaceModel.Text.Bert_Tiny2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tr = TextRecognizer(model=model, device=device)

tr.recognize('это работает? :(', return_single_label=True)
```
#### Распознавание эмоций в голосе

```shell
# @title Text: wavlm-emotion-russian-resd 
import torch
from aniemore.recognizers.voice import VoiceRecognizer
from aniemore.models import HuggingFaceModel

model=HuggingFaceModel.Voice.WavLM
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vr = VoiceRecognizer(model=model, device=device)
vr.recognize('/content/ваш-звуковой-файл.wav', return_single_label=True)
```
<hr>

## Доп. ссылки

Все модели и датасеты, а так же примеры их использования вы можете посмотреть в нашем [HuggingFace профиле](https://huggingface.co/Aniemore)

## Аффилированость
**Aniemore (Artem Nikita Ilya EMOtion REcognition)**

Разработка открытой библиотеки произведена коллективом авторов на базе ООО "Социальный код".
Результаты работы получены за счет гранта Фонда содействия развитию малых форм предприятий в научно-технической сфере (Договор №1ГУКодИИС12-D7/72697
от 22.12.2021).

## Цитирование
```
@misc{Aniemore,
  author = {Артем Аментес, Илья Лубенец, Никита Давидчук},
  title = {Открытая библиотека искусственного интеллекта для анализа и выявления эмоциональных оттенков речи человека},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/aniemore/Aniemore}},
  email = {hello@socialcode.ru}
}
```
