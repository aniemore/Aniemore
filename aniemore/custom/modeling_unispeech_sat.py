"""Base model classes
"""
from aniemore.custom.modeling_classificators import BaseModelForVoiceBaseClassification
from transformers import UniSpeechSatForSequenceClassification


class UniSpeechSatForVoiceClassification(BaseModelForVoiceBaseClassification):
    """UniSpeechSatForVoiceClassification is a model for voice classification task
     (e.g. speech command, voice activity detection, etc.)

    Args:
        config (UniSpeechSatConfig): config
        num_labels (int): number of labels

    Attributes:
        config (UniSpeechSatConfig): config
        num_labels (int): number of labels
        unispeech_sat (UniSpeechSatForSequenceClassification): unispeech_sat model
    """

    def __init__(self, config, num_labels):
        super().__init__(config, num_labels)
        self.unispeech_sat = UniSpeechSatForSequenceClassification(config)
        self.init_weights()
