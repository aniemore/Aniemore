from typing import List

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from Aniemore.Utils import s2t
from Aniemore.Voice import Wav2Vec2ForSpeechClassification
from Aniemore.config import config
from Aniemore.Utils import MasterModel
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Processor


# > This class takes in a .wav file and returns the emotion of the speaker
class EmotionFromVoice(MasterModel):
    MODEL_URL = config["Huggingface"]["models"]["wav2vec2_53_voice"]
    TRC = True
    SAMPLE_RATE = config["Voice"]["preprocess"]["audio-default-sample"]

    model_config: Wav2Vec2Config = None
    feature_extractor: Wav2Vec2FeatureExtractor = None
    processor: Wav2Vec2Processor = None
    model: Wav2Vec2ForSpeechClassification = None

    def __init__(self):
        super().__init__()

    def setup_variables(self):
        """
        Он устанавливает переменные, которые будут использоваться в программе.
        """
        self.model_config = Wav2Vec2Config.from_pretrained(self.MODEL_URL, trust_remote_code=self.TRC)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.MODEL_URL)
        self.processor = Wav2Vec2Processor.from_pretrained(self.MODEL_URL)
        self.model = Wav2Vec2ForSpeechClassification.from_pretrained(self.MODEL_URL, config=self.model_config)

    @staticmethod
    def speech_file_to_array_fn(path):
        """
        Он берет путь к файлу .wav, считывает его и возвращает пустой массив аудиоданных.

        :param path: путь к файлу
        """
        speech_array, _sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def _predict_one(self, path: str, single_label) -> List[dict] or List[str]:
        """
        Он берет путь к изображению и возвращает список словарей, каждый из которых содержит имя класса и вероятность того,
        что изображение принадлежит этому классу.

        :param path: Путь к предсказываемому изображению
        :type path: str
        """
        speech = self.speech_file_to_array_fn(path)
        inputs = self.feature_extractor(speech, sampling_rate=self.SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to(self.device) for key in inputs}

        with torch.no_grad():
            logits = self.model.to(self.device)(**inputs).logits

        scores = F.softmax(logits, dim=1).detach().cpu()

        if single_label is False:
            scores = scores.numpy()[0]
            outputs = [{self.model_config.id2label[i]: v for i, v in enumerate(scores)}]

        else:
            max_score = torch.argmax(scores, dim=1).numpy()
            outputs = [self.model_config.id2label[max_score[0]]]

        return outputs

    def _predict_many(self, paths: List[str], single_label) -> List[List[dict]] or List[List[str]]:
        """
        Он принимает список путей к изображениям и возвращает список прогнозов для каждого изображения.

        :param paths: Список[стр]
        :type paths: List[str]
        """
        speeches = []

        for _path in paths:
            speech = self.speech_file_to_array_fn(_path)
            speeches.append(speech)

        features = self.processor(speeches, sampling_rate=self.processor.feature_extractor.sampling_rate,
                             return_tensors="pt", padding=True)

        input_values = features.input_values.to(self.device)
        attention_mask = features.attention_mask.to(self.device)

        with torch.no_grad():
            logits = self.model.to(self.device)(input_values, attention_mask=attention_mask).logits

        scores = F.softmax(logits, dim=1).detach().cpu().numpy()

        outputs = []

        for _local_path, _local_score in zip(paths, scores):
            if single_label is False:
                outputs.append(
                    [_local_path, {self.model_config.id2label[i]: v for i, v in enumerate(_local_score)}]
                )

            else:
                max_score = np.argmax(_local_score)
                outputs.append(
                    [_local_path, self.model_config.id2label[max_score]]
                )

        return outputs

    def predict(self, path: List[str] or str, single_label=False) -> List[dict] or List[List[dict]] or\
                                                                     List[str] or List[List[str]]:
        """
        > Эта функция принимает путь к файлу или список путей к файлам и возвращает список словарей или список списков
        словарей

        :param path: Путь к изображению, которое вы хотите предсказать
        :type path: List[str] or str
        """
        if self.model is None:
            self.setup_variables()

        if type(path) == str:
            return self._predict_one(path, single_label=single_label)

        elif type(path) == list:
            return self._predict_many(path, single_label=single_label)

        else:
            raise ValueError("You need to input list[paths] or one path of your file for prediction")
