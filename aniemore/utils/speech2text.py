import whisper
from typing import NamedTuple, List, Dict


class SpeechSegment(NamedTuple):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float


class Speech2TextOutput(NamedTuple):
    text: str
    language: str
    segments: List[SpeechSegment]


class Speech2Text:
    def __init__(self, model_path: str):
        self.model = whisper.load_model(model_path)

    def __call__(self, audio_path: str) -> Speech2TextOutput:
        return self.recognize(audio_path)

    def recognize(self, audio_path: str) -> Speech2TextOutput:
        result = self.model.transcribe(audio_path)
        result['segments'] = [SpeechSegment(**x) for x in result['segments']]
        return Speech2TextOutput(**result)


if __name__ == '__main__':
    model_path = 'small'
    audio_path = '../recognizer/voice.wav'

    recognizer = Speech2Text(model_path)
    print(recognizer(audio_path).segments[0])
    print(recognizer.recognize(audio_path))
