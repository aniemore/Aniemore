from dataclasses import dataclass
import torch
from transformers.utils import ModelOutput
from transformers import (
    Wav2Vec2ForSequenceClassification,
    WavLMForSequenceClassification,
    UniSpeechSatForSequenceClassification,
    HubertForSequenceClassification,
    PreTrainedModel
)


@dataclass
class SpeechModelOutput(ModelOutput):
    loss: torch.FloatTensor
    logits: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None


class BaseModelForVoiceClassification(PreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config=config)
        self.num_labels = num_labels
        self.pooling_mode = config.pooling_mode
        self.projector = torch.nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = torch.nn.Linear(config.classifier_proj_size, config.num_labels)

    @staticmethod
    def merged_strategy(
            hidden_states,
            mode="mean"
    ):
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

    def compute_loss(self, logits, labels):
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

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
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


class Wav2Vec2ForVoiceClassification(BaseModelForVoiceClassification):
    def __init__(self, config, num_labels):
        super().__init__(config, num_labels)
        self.wav2vec2 = Wav2Vec2ForSequenceClassification(config)
        self.init_weights()


class WavLMForVoiceClassification(BaseModelForVoiceClassification):
    def __init__(self, config, num_labels):
        super().__init__(config, num_labels)
        self.wavlm = WavLMForSequenceClassification(config)
        self.init_weights()


class UniSpeechSatForVoiceClassification(BaseModelForVoiceClassification):
    def __init__(self, config, num_labels):
        super().__init__(config, num_labels)
        self.unispeech_sat = UniSpeechSatForSequenceClassification(config)
        self.init_weights()


class HubertForVoiceClassification(BaseModelForVoiceClassification):
    def __init__(self, config, num_labels):
        super().__init__(config, num_labels)
        self.hubert = HubertForSequenceClassification(config)
        self.init_weights()
