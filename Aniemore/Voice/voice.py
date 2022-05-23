import torch
import torchaudio
import torch.nn.functional as F
from Aniemore.Utils import s2t
from Aniemore.Voice import Wav2Vec2ForSpeechClassification
from Aniemore.config import config
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Processor


# > This class takes in a .wav file and returns the emotion of the speaker
class EmotionFromVoice:
    MODEL_URL = config["Huggingface"]["wav2vec2_53_voice"]
    TRC = True

    model_config = Wav2Vec2Config.from_pretrained(MODEL_URL, trust_remote_code=TRC)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_URL)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_URL)
    model = Wav2Vec2ForSpeechClassification.from_pretrained(MODEL_URL, config=model_config)

    def __init__(self):
        self.device = None

    def to(self, device):
        if type(device) == str:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        elif type(device) == torch.device:
            self.device = device

        else:
            raise ValueError("Unknown acceleration device")

    @torch.no_grad()
    def speech_file_to_array_fn_(self, batch):
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        speech_array = speech_array.squeeze().numpy()
        speech_array = librosa.resample(np.asarray(speech_array), sampling_rate,
                                        processor.feature_extractor.sampling_rate)

        batch["speech"] = speech_array
        return batch

    def predict_(self, batch):
        features = self.processor(batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate,
                             return_tensors="pt", padding=True)

        input_values = features.input_values.to(self.device)
        attention_mask = features.attention_mask.to(self.device)

        with torch.no_grad():
            logits = model(input_values, attention_mask=attention_mask).logits

        pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        batch["predicted"] = pred_ids
        batch["default_predicted"] = logits.detach().cpu().numpy()
        return batch

    def speech_file_to_array_fn(self, path):
        speech_array, _sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def predict(self, path, sampling_rate):
        speech = self.speech_file_to_array_fn(path, sampling_rate)
        inputs = self.feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {key: inputs[key].to(self.device) for key in inputs}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            print(logits)

        scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in
                   enumerate(scores)]
        return outputs

