#from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F
from transformers import DistilBertPreTrainedModel
from transformers import DistilBertConfig


class DistilBert(DistilBertPreTrainedModel):
    def __init__(self, config, projection=False):
        super(DistilBert, self).__init__(config)

        self.distilbert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.projection = projection
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        if self.projection:
            self.projection_layer = nn.Linear(config.dim,256)
            self.classifier = nn.Linear(256,config.num_labels)
        else:
            self.classifier = nn.Linear(config.dim, config.num_labels)
        #print(config.num_labels)

    def forward(self, input, start_layer_idx = 0):
        if start_layer_idx >= 0:
            input_ids, attention_mask = input
            distilbert_output = self.distilbert(input_ids=input_ids,
                                                attention_mask=attention_mask)

            hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
            pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
            pooled_output = self.dropout(pooled_output)  # (bs, dim)

            if self.projection:
                pooled_output = self.projection_layer(pooled_output)
        else:
            pooled_output = input
        logits = self.classifier(pooled_output)  # (bs, dim)

        # Convert logits → log probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (bs, num_labels)
        return log_probs #pooled_output,logits
    
    def get_logits(self, input, start_layer_idx = 0):
        if start_layer_idx >= 0:
            input_ids, attention_mask = input
            distilbert_output = self.distilbert(input_ids=input_ids,
                                                attention_mask=attention_mask)

            hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
            pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
            pooled_output = self.dropout(pooled_output)  # (bs, dim)

            if self.projection:
                pooled_output = self.projection_layer(pooled_output)
        else:
            pooled_output = input
        logits = self.classifier(pooled_output)  # (bs, dim)

        return logits #pooled_output,logits


def distilbert():

    bert_config = DistilBertConfig(num_labels=4)
    model = DistilBert(bert_config)

    return model

