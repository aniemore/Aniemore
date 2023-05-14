"""Base model classes
"""
from dataclasses import dataclass
from typing import Union, Type, Tuple

import torch
from huggingface_hub import PyTorchModelHubMixin
from transformers.utils import ModelOutput
from transformers import (
    PreTrainedModel,
    PretrainedConfig
)


@dataclass
class SpeechModelOutput(ModelOutput):
    """Base class for model's outputs, with potential hidden states and attentions.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of
            each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attention's weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.

    Examples::
        >>> from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Tokenizer
        >>> import torch
        >>>
        >>> tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        >>> model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h")
        >>> input_values = tokenizer("Hello, my dog is cute", return_tensors="pt").input_values  # Batch size 1
        >>> logits = model(input_values).logits
        >>> assert logits.shape == (1, 2)
    """
    loss: torch.FloatTensor
    logits: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None


class MultiModalConfig(PretrainedConfig):
    """Base class for multimodal configs"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BaseClassificationModel(PreTrainedModel, PyTorchModelHubMixin):
    config: Type[Union[PretrainedConfig, None]] = None

    def compute_loss(self, logits, labels):
        """Compute loss

        Args:
            logits (torch.FloatTensor): logits
            labels (torch.LongTensor): labels

        Returns:
            torch.FloatTensor: loss

        Raises:
            ValueError: Invalid number of labels
        """
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1:
                self.config.problem_type = "single_label_classification"
            else:
                raise ValueError("Invalid number of labels: {}".format(self.num_labels))

        if self.config.problem_type == "single_label_classification":
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        elif self.config.problem_type == "multi_label_classification":
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

        elif self.config.problem_type == "regression":
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            raise ValueError("Problem_type {} not supported".format(self.config.problem_type))

        return loss

    @staticmethod
    def merged_strategy(
            hidden_states,
            mode="mean"
    ):
        """Merged strategy for pooling

        Args:
            hidden_states (torch.FloatTensor): hidden states
            mode (str, optional): pooling mode. Defaults to "mean".

        Returns:
            torch.FloatTensor: pooled hidden states
        """
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def get_position_embeddings(self) -> Union[torch.nn.Embedding, Tuple[torch.nn.Embedding]]:
        pass

    def prepare_inputs_for_generation(self, *args, **kwargs):
        pass

    def _reorder_cache(self, past_key_values, beam_idx):
        pass


class BaseModelForVoiceBaseClassification(BaseClassificationModel):
    def __init__(self, config, num_labels):
        """Base model for voice classification

        Args:
            config (PretrainedConfig): config
            num_labels (int): number of labels
        """
        super().__init__(config=config)
        self.num_labels = num_labels
        self.pooling_mode = config.pooling_mode
        self.projector = torch.nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = torch.nn.Linear(config.classifier_proj_size, config.num_labels)

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        """Forward

        Args:
            input_values (torch.FloatTensor): input values
            attention_mask (torch.LongTensor, optional): attention mask. Defaults to None.
            output_attentions (bool, optional): output attentions. Defaults to None.
            output_hidden_states (bool, optional): output hidden states. Defaults to None.
            return_dict (bool, optional): return dict. Defaults to None.
            labels (torch.LongTensor, optional): labels. Defaults to None.

        Returns:
            torch.FloatTensor: logits

        Raises:
            ValueError: Invalid number of labels
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = self.projector(outputs.last_hidden_state)
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BaseMultiModalForSequenceBaseClassification(BaseClassificationModel):
    config_class = MultiModalConfig

    def __init__(self, config):
        """
        Args:
            config (MultiModalConfig): config

        Attributes:
            config (MultiModalConfig): config
            num_labels (int): number of labels
            audio_config (Union[PretrainedConfig, None]): audio config
            text_config (Union[PretrainedConfig, None]): text config
            audio_model (Union[PreTrainedModel, None]): audio model
            text_model (Union[PreTrainedModel, None]): text model
            classifier (Union[torch.nn.Linear, None]): classifier
        """
        super().__init__(config)
        self.config = config
        self.num_labels = self.config.num_labels
        self.audio_config: Union[PretrainedConfig, None] = None
        self.text_config: Union[PretrainedConfig, None] = None
        self.audio_model: Union[PreTrainedModel, None] = None
        self.text_model: Union[PreTrainedModel, None] = None
        self.classifier: Union[torch.nn.Linear, None] = None

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
        audio_mean = self.merged_strategy(audio_output.last_hidden_state, mode=self.config.pooling_mode)

        pooled_output = torch.cat(
            (audio_mean, text_output.pooler_output), dim=1
        )
        logits = self.classifier(pooled_output)
        loss = None

        if labels is not None:
            loss = self.compute_loss(logits, labels)

        return SpeechModelOutput(
            loss=loss,
            logits=logits
        )


class AudioTextFusionModelForSequenceClassificaion(BaseMultiModalForSequenceBaseClassification):
    def __init__(self, config):
        """
        Args:
            config (MultiModalConfig): config
        Attributes:
            audio_projector (Union[torch.nn.Linear, None]): Projection layer for audio embeds
            text_projector (Union[torch.nn.Linear, None]): Projection layer for text embeds
            audio_avg_pool (Union[torch.nn.AvgPool1d, None]): Audio average pool (out from fusion block)
            text_avg_pool (Union[torch.nn.AvgPool1d, None]): Text average pool (out from fusion block)
        """
        super().__init__(config)
        self.audio_projector: Union[torch.nn.Linear, None] = None
        self.text_projector: Union[torch.nn.Linear, None] = None
        self.audio_avg_pool: Union[torch.nn.AvgPool1d, None] = None
        self.text_avg_pool: Union[torch.nn.AvgPool1d, None] = None
