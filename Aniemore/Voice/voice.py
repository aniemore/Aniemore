from typing import List

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from Aniemore.Utils import s2t
from Aniemore.Voice import Wav2Vec2ForSpeechClassification
from Aniemore.config import config
from Aniemore.Utils import MasterModel
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Processor


# > This class takes in a .wav file and returns the emotion of the speaker
class EmotionFromVoice(MasterModel):
    MODEL_URL = config["Huggingface"]["wav2vec2_53_voice"]
    TRC = True
    SAMPLE_RATE = 16000

    model_config: Wav2Vec2Config = None
    feature_extractor: Wav2Vec2FeatureExtractor = None
    processor: Wav2Vec2Processor = None
    model: Wav2Vec2ForSpeechClassification = None

    def __init__(self):
        super().__init__()

        self.device = None

    def to(self, device):
        """
        If the device is a string, then it will be converted to a torch.device object. If the device is a torch.device
        object, then it will be assigned to self.device. If the device is neither a string nor a torch.device object, then
        an error will be raised

        :param device: The device to run the model on
        """
        if type(device) == str:
            self.device = {
                "cuda": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                "cpu": torch.device("cpu")
            }.get(device, torch.device("cpu"))
            return self

        elif type(device) == torch.device:
            self.device = device
            return self

        else:
            raise ValueError("Unknown acceleration device")

    def setup_variables(self):
        """
        We're loading the model, the feature extractor, and the processor from the model URL
        """
        self.model_config = Wav2Vec2Config.from_pretrained(self.MODEL_URL, trust_remote_code=self.TRC)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.MODEL_URL)
        self.processor = Wav2Vec2Processor.from_pretrained(self.MODEL_URL)
        self.model = Wav2Vec2ForSpeechClassification.from_pretrained(self.MODEL_URL, config=self.model_config)

    @staticmethod
    def speech_file_to_array_fn(path):
        """
        It takes a path to a .wav file, loads it, resamples it to 16kHz, and returns the audio as a numpy array

        :param path: the path to the audio file
        :return: The speech array is being returned.
        """
        speech_array, _sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def _predict_one(self, path: str) -> List[dict]:
        """
        > We take the audio file, convert it to a tensor, pass it through the model, and return the output

        :param path: The path to the audio file to be predicted
        :type path: str
        :return: A list of dictionaries.
        """
        speech = self.speech_file_to_array_fn(path)
        inputs = self.feature_extractor(speech, sampling_rate=self.SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to(self.device) for key in inputs}

        with torch.no_grad():
            logits = self.model.to(self.device)(**inputs).logits

        scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        outputs = [{self.model_config.id2label[i]: v for i, v in enumerate(scores)}]

        return outputs

    def _predict_many(self, paths: List[str]) -> List[list[str, dict]]:
        """
        > We take a list of paths to audio files, convert them to arrays, pass them through the model, and return a list of
        dictionaries with the probabilities of each emotion

        :param paths: a list of paths to the audio files you want to predict on
        :type paths: List[str]
        :return: A list of lists, where each list contains the path to the audio file and a dictionary of the scores for
        each label.
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
            outputs.append(
                [_local_path, {self.model_config.id2label[i]: v for i, v in enumerate(_local_score)}]
            )

        return outputs

    def predict(self, path: List[str] or str) -> List[dict] or List[list[str, dict]]:
        """
        If the model is not set up, set it up. If the path is a string, predict one file. If the path is a list, predict
        many files

        :param path: The path to the file you want to predict
        :type path: List[str] or str
        :return: A list of dictionaries or a list of lists of strings and dictionaries.
        """
        if self.model is None:
            self.setup_variables()

        if type(path) == str:
            return self._predict_one(path)

        elif type(path) == list:
            return self._predict_many(path)

        else:
            raise ValueError("You need to input list[paths] or one path of your file for prediction")
