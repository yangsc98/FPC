import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast

from transformers import RobertaConfig, RobertaPreTrainedModel, RobertaModel
from transformers.activations import gelu


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


class RobertaForMaskedLM(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        roberta_outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        sequence_output = roberta_outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return (masked_lm_loss, prediction_scores, sequence_output)


class robertaMLMRE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.config = RobertaConfig.from_pretrained(self.args.model_name_or_path)
        if self.args.use_gradient_checkpoint:
            self.config.gradient_checkpointing = True

        self.robertaMLM = RobertaForMaskedLM.from_pretrained(self.args.model_name_or_path, config=self.config)

        self.relation_cls = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(self.config.hidden_size, self.args.label_num)
        )

    @autocast()
    def forward(
        self,
        input_ids,
        attention_mask,
        subj_start,
        obj_start,
        MLM_labels=None,
        cls_label=None,
        mlm_ratio=None,
    ):
        """
        Input:
            input_ids:                  [batch_size, seq_len]
            attention_mask:             [batch_size, seq_len]
            subj_start:                 [batch_size]
            obj_start:                  [batch_size]
            MLM_labels:                 [batch_size, seq_len]
            cls_label:                  [batch_size]

        Output:
            loss:                       []
            cls_logits:                 [batch_size, label_num]
            MLM_logits:                 [batch_size, seq_len, vocab_size]
        """

        if not mlm_ratio:
            mlm_ratio = self.args.mlm_ratio

        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        robertaMLM_outputs = self.robertaMLM(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=MLM_labels,
        )
        MLM_loss, MLM_logits, sequence_output = robertaMLM_outputs

        index_range = torch.arange(batch_size).to(device)
        head_output = sequence_output[index_range, subj_start]
        tail_output = sequence_output[index_range, obj_start]

        relation_output = torch.cat((head_output, tail_output), 1)    # [batch_size, hidden_size * 2]
        cls_logits = self.relation_cls(relation_output)    # [batch_size, label_num]

        loss = None
        if cls_label is not None:
            loss_fct = CrossEntropyLoss()
            cls_loss = loss_fct(cls_logits.view(-1, self.args.label_num), cls_label.view(-1))

            if MLM_loss is not None:
                loss = mlm_ratio * MLM_loss + (1 - mlm_ratio) * cls_loss
            else:
                loss = cls_loss

        outputs = {
            "loss": loss,
            "cls_logits": cls_logits,
            "MLM_logits": MLM_logits,
        }

        return outputs
