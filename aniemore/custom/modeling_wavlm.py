"""Base model classes
"""
from typing import Union, Dict

from aniemore.custom.modeling_classificators import (
    BaseModelForVoiceBaseClassification,
    BaseMultiModalForSequenceBaseClassification,
    AudioTextFusionModelForSequenceClassificaion,
    SpeechModelOutput,
    MultiModalConfig
)

import torch
from transformers import (
    WavLMForSequenceClassification,
    WavLMConfig,
    BertConfig,
    WavLMModel,
    BertModel,
    PretrainedConfig,
)

from transformers.models.wavlm.modeling_wavlm import (
    WavLMEncoder,
    WavLMEncoderStableLayerNorm,
    WavLMFeatureEncoder
)

from transformers.models.bert.modeling_bert import BertEncoder


class FusionConfig(MultiModalConfig):
    """Base class for fusion configs
    Just for fine-tuning models, no more

    Args:
        audio_config (PretrainedConfig): audio config
        text_config (PretrainedConfig): text config
        id2label (Dict[int, str]): id2label
        label2id (Dict[str, int]): label2id
        num_heads (int, optional): number of heads. Defaults to 8.
        kernel_size (int, optional): kernel size. Defaults to 1.
        pooling_mode (str, optional): pooling mode. Defaults to "mean".
        problem_type (str, optional): problem type. Defaults to "single_label_classification".
        gradient_checkpointing (bool, optional): gradient checkpointing. Defaults to True.
    """

    def __init__(
            self,
            audio_config: PretrainedConfig,
            text_config: PretrainedConfig,
            id2label: Dict[int, str],
            label2id: Dict[str, int],
            num_heads: int = 8,
            kernel_size: int = 1,
            pooling_mode: str = "mean",
            problem_type: str = "single_label_classification",
            gradient_checkpointing: bool = True,
            **kwargs):
        super().__init__(**kwargs)

        self.update({audio_config.architectures[0]: audio_config.to_dict()})
        self.update({text_config.architectures[0]: text_config.to_dict()})

        self.id2label = id2label
        self.label2id = label2id
        self.num_labels = len(id2label)
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.pooling_mode = pooling_mode
        self.problem_type = problem_type
        self.gradient_checkpointing = gradient_checkpointing


class FusionModuleQ(torch.nn.Module):
    """FusionModuleQ is a fusion module for the query
    https://arxiv.org/abs/2302.13661
    https://arxiv.org/abs/2207.04697

    Args:
        audio_dim (int): audio dimension
        text_dim (int): text dimension
        num_heads (int): number of heads
    """

    def __init__(self, audio_dim, text_dim, num_heads):
        super().__init__()

        # pick the lowest dimension of the two modalities
        self.dimension = min(audio_dim, text_dim)

        # attention modules
        self.a_self_attention = torch.nn.MultiheadAttention(self.dimension, num_heads=num_heads)
        self.t_self_attention = torch.nn.MultiheadAttention(self.dimension, num_heads=num_heads)

        # layer norm
        self.audio_norm = torch.nn.LayerNorm(self.dimension)
        self.text_norm = torch.nn.LayerNorm(self.dimension)


class WavLMForVoiceClassification(BaseModelForVoiceBaseClassification):
    """WavLMForVoiceClassification is a model for voice classification task
     (e.g. speech command, voice activity detection, etc.)

    Args:
        config (WavLMConfig): config
        num_labels (int): number of labels

    Attributes:
        config (WavLMConfig): config
        num_labels (int): number of labels
        wavlm (WavLMForSequenceClassification): wavlm model
    """

    def __init__(self, config, num_labels):
        super().__init__(config, num_labels)
        self.wavlm = WavLMForSequenceClassification(config)
        self.init_weights()


class WavLMBertForSequenceClassification(BaseMultiModalForSequenceBaseClassification):
    """WavLMBertForSequenceClassification is a model for sequence classification task
     (e.g. sentiment analysis, text classification, etc.)

    Args:
        config (WavLMBertConfig): config

    Attributes:
        config (WavLMBertConfig): config
        audio_config (WavLMConfig): wavlm config
        text_config (BertConfig): bert config
        audio_model (WavLMModel): wavlm model
        text_model (BertModel): bert model
        classifier (torch.nn.Linear): classifier
    """

    def __init__(self, config, finetune=False):
        super().__init__(config)
        self.supports_gradient_checkpointing = getattr(config, "gradient_checkpointing", True)

        self.audio_config = WavLMConfig.from_dict(self.config.WavLMModel)
        self.text_config = BertConfig.from_dict(self.config.BertModel)

        if not finetune:
            self.audio_model = WavLMModel(self.audio_config)
            self.text_model = BertModel(self.text_config)

        else:
            self.audio_model = WavLMModel.from_pretrained(self.audio_config._name_or_path, config=self.audio_config)
            self.text_model = BertModel.from_pretrained(self.text_config._name_or_path, config=self.text_config)

        self.classifier = torch.nn.Linear(
            self.audio_config.hidden_size + self.text_config.hidden_size, self.num_labels
        )
        self.init_weights()

    @staticmethod
    def _set_gradient_checkpointing(module, value=False):
        if isinstance(module, (WavLMEncoder, WavLMEncoderStableLayerNorm, WavLMFeatureEncoder, BertEncoder)):
            module.gradient_checkpointing = value


class WavLMBertFusionForSequenceClassification(AudioTextFusionModelForSequenceClassificaion):
    """WavLMBertForSequenceClassification is a model for sequence classification task
     (e.g. sentiment analysis, text classification, etc.) for fine-tuning
    Args:
        config (WavLMBertConfig): config
    Attributes:
        config (WavLMBertConfig): config
        audio_config (WavLMConfig): wavlm config
        text_config (BertConfig): bert config
        audio_model (WavLMModel): wavlm model
        text_model (BertModel): bert model
        fusion_module_{i} (FusionModuleQ): Fusion Module Q
        audio_projector (Union[torch.nn.Linear, None]): Projection layer for audio embeds
        text_projector (Union[torch.nn.Linear, None]): Projection layer for text embeds
        audio_avg_pool (Union[torch.nn.AvgPool1d, None]): Audio average pool (out from fusion block)
        text_avg_pool (Union[torch.nn.AvgPool1d, None]): Text average pool (out from fusion block)
        classifier (torch.nn.Linear): classifier
    """

    def __init__(self, config, finetune=False):
        super().__init__(config)
        self.supports_gradient_checkpointing = getattr(config, "gradient_checkpointing", True)

        self.audio_config = WavLMConfig.from_dict(self.config.WavLMModel)
        self.text_config = BertConfig.from_dict(self.config.BertModel)

        if not finetune:
            self.audio_model = WavLMModel(self.audio_config)
            self.text_model = BertModel(self.text_config)

        else:
            self.audio_model = WavLMModel.from_pretrained(self.audio_config._name_or_path, config=self.audio_config)
            self.text_model = BertModel.from_pretrained(self.text_config._name_or_path, config=self.text_config)

        # fusion module with V3 strategy (one projection on entry, no projection in continuous)
        for i in range(self.config.num_fusion_layers):
            setattr(self, f"fusion_module_{i + 1}", FusionModuleQ(
                self.audio_config.hidden_size, self.text_config.hidden_size, self.config.num_heads
            ))

        self.audio_projector = torch.nn.Linear(self.audio_config.hidden_size, self.text_config.hidden_size)
        self.text_projector = torch.nn.Linear(self.text_config.hidden_size, self.text_config.hidden_size)

        # Avg Pool
        self.audio_avg_pool = torch.nn.AvgPool1d(self.config.kernel_size)
        self.text_avg_pool = torch.nn.AvgPool1d(self.config.kernel_size)

        # output dimensions of wav2vec2 and bert are 768 and 1024 respectively
        cls_dim = min(self.audio_config.hidden_size, self.text_config.hidden_size)
        self.classifier = torch.nn.Linear(
            (cls_dim * 2) // self.config.kernel_size, self.config.num_labels
        )

        self.init_weights()

    @staticmethod
    def _set_gradient_checkpointing(module, value=False):
        if isinstance(module, (WavLMEncoder, WavLMEncoderStableLayerNorm, WavLMFeatureEncoder, BertEncoder)):
            module.gradient_checkpointing = value

    def forward(
            self,
            input_ids=None,
            input_values=None,
            text_attention_mask=None,
            audio_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
    ):
        """Forward method for multimodal model for sequence classification task (e.g. text + audio)
        Args:
            input_ids (torch.LongTensor, optional): input ids. Defaults to None.
            input_values (torch.FloatTensor, optional): input values. Defaults to None.
            text_attention_mask (torch.LongTensor, optional): text attention mask. Defaults to None.
            audio_attention_mask (torch.LongTensor, optional): audio attention mask. Defaults to None.
            token_type_ids (torch.LongTensor, optional): token type ids. Defaults to None.
            position_ids (torch.LongTensor, optional): position ids. Defaults to None.
            head_mask (torch.FloatTensor, optional): head mask. Defaults to None.
            inputs_embeds (torch.FloatTensor, optional): inputs embeds. Defaults to None.
            labels (torch.LongTensor, optional): labels. Defaults to None.
            output_attentions (bool, optional): output attentions. Defaults to None.
            output_hidden_states (bool, optional): output hidden states. Defaults to None.
            return_dict (bool, optional): return dict. Defaults to True.
        Returns:
            torch.FloatTensor: logits
        """
        audio_output = self.audio_model(
            input_values=input_values,
            attention_mask=audio_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=text_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Mean pooling
        audio_avg = self.merged_strategy(audio_output.last_hidden_state, mode=self.config.pooling_mode)

        # Projection
        audio_proj = self.audio_projector(audio_avg)
        text_proj = self.text_projector(text_output.pooler_output)

        audio_mha, text_mha = None, None

        for i in range(self.config.num_fusion_layers):
            fusion_module = getattr(self, f"fusion_module_{i + 1}")

            if i == 0:
                audio_mha, text_mha = fusion_module(audio_proj, text_proj)
            else:
                audio_mha, text_mha = fusion_module(audio_mha, text_mha)

        audio_avg = self.audio_avg_pool(audio_mha)
        text_avg = self.text_avg_pool(text_mha)

        fusion_output = torch.concat((audio_avg, text_avg), dim=1)

        logits = self.classifier(fusion_output)
        loss = None

        if labels is not None:
            loss = self.compute_loss(logits, labels)

        return SpeechModelOutput(
            loss=loss,
            logits=logits
        )
