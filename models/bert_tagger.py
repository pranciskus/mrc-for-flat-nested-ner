#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.04.23 
# First create: 2019.04.23 
# Description:
# bert_tagger.py 


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss 


from layers.classifier import * 
from layers.bert_basic_model import * 
from layers.bert_layernorm import BertLayerNorm 




class BertTagger(nn.Module):
    def __init__(self, config, num_labels=5):
        super(BertTagger, self).__init__()
        self.num_labels = 5 

        bert_config = BertConfig.from_dict(config.bert_config.to_dict())
        self.bert = BertModel(bert_config)

        self.hidden_size = config.hidden_size 
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = MultiNonLinearClassifier(config.hidden_size, self.num_labels)
        self.bert = self.bert.from_pretrained(config.bert_model, )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
        labels=None, input_mask=None):

        last_bert_layer, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, \
            output_all_encoded_layers=False)
        last_bert_layer = last_bert_layer.view(-1, self.hidden_size)
        last_bert_layer = self.dropout(last_bert_layer)
        logits = self.classifier(last_bert_layer) 

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if input_mask is not None:
                masked_logits = torch.masked_select(logits, input_mask)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) 
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss 
        else:
            return logits 


