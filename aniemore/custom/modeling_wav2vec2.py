"""Base model classes
"""
from aniemore.custom.modeling_classificators import (
    BaseModelForVoiceBaseClassification,
    BaseMultiModalForSequenceBaseClassification
)

import torch
from transformers import (
    Wav2Vec2ForSequenceClassification,
    BertConfig,
    BertModel,
    Wav2Vec2Config,
    Wav2Vec2Model
)

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Encoder,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2FeatureEncoder
)

from transformers.models.bert.modeling_bert import BertEncoder


class Wav2Vec2ForVoiceClassification(BaseModelForVoiceBaseClassification):
    """Wav2Vec2ForVoiceClassification is a model for voice classification task
     (e.g. speech command, voice activity detection, etc.)

    Args:
        config (Wav2Vec2Config): config
        num_labels (int): number of labels

    Attributes:
        config (Wav2Vec2Config): config
        num_labels (int): number of labels
        wav2vec2 (Wav2Vec2ForSequenceClassification): wav2vec2 model
    """

    def __init__(self, config, num_labels):
        super().__init__(config, num_labels)
        self.wav2vec2 = Wav2Vec2ForSequenceClassification(config)
        self.init_weights()


class Wav2Vec2BertForSequenceClassification(BaseMultiModalForSequenceBaseClassification):
    """Wav2Vec2BertForSequenceClassification is a model for sequence classification task
     (e.g. sentiment analysis, text classification, etc.)

    Args:
        config (Wav2Vec2BertConfig): config

    Attributes:
        config (Wav2Vec2BertConfig): config
        audio_config (Wav2Vec2Config): wav2vec2 config
        text_config (BertConfig): bert config
        audio_model (Wav2Vec2Model): wav2vec2 model
        text_model (BertModel): bert model
        classifier (torch.nn.Linear): classifier
    """

    def __init__(self, config, finetune=False):
        super().__init__(config)
        self.supports_gradient_checkpointing = getattr(config, "gradient_checkpointing", True)

        self.audio_config = Wav2Vec2Config.from_dict(self.config.Wav2Vec2Model)
        self.text_config = BertConfig.from_dict(self.config.BertModel)

        if not finetune:
            self.audio_model = Wav2Vec2Model(self.audio_config)
            self.text_model = BertModel(self.text_config)

        else:
            self.audio_model = Wav2Vec2Model.from_pretrained(self.audio_config._name_or_path, config=self.audio_config)
            self.text_model = BertModel.from_pretrained(self.text_config._name_or_path, config=self.text_config)

        self.classifier = torch.nn.Linear(
            self.audio_config.hidden_size + self.text_config.hidden_size, self.num_labels
        )
        self.init_weights()

    @staticmethod
    def _set_gradient_checkpointing(module, value=False):
        if isinstance(module, (Wav2Vec2Encoder, Wav2Vec2EncoderStableLayerNorm, Wav2Vec2FeatureEncoder, BertEncoder)):
            module.gradient_checkpointing = value
