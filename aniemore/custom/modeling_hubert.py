"""Base model classes
"""
from aniemore.custom.modeling_classificators import BaseModelForVoiceBaseClassification
from transformers import HubertForSequenceClassification


class HubertForVoiceClassification(BaseModelForVoiceBaseClassification):
    """HubertForVoiceClassification is a model for voice classification task
     (e.g. speech command, voice activity detection, etc.)

    Args:
        config (HubertConfig): config
        num_labels (int): number of labels

    Attributes:
        config (HubertConfig): config
        num_labels (int): number of labels
        hubert (HubertForSequenceClassification): hubert model
    """

    def __init__(self, config, num_labels):
        super().__init__(config, num_labels)
        self.hubert = HubertForSequenceClassification(config)
        self.init_weights()
