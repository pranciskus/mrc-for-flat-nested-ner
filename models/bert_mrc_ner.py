#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.04.28 
# First create: 2019.03.23 
# Description:
# Description:
# Bert Model for Query-Based NER Task
# ---------------------------------------------------------------------------
# There exists differences between Query-NER and QA. 
# 1. may exists multi-span in a passage. 
#       so we choose Sigmoid instead of softmax as activ func. 
# ----------------------------------------------------------------------------


import os 
import sys 
import copy 
import json 
import math 
import logging 
import numpy as np 



root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print("the root_path of current file is: ")
print(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss 


from layers.bert_basic_model import BertModel, PreTrainedBertModel, BertConfig  



class BertMRCNER(nn.Module):
    """
    Desc:
        BERT model for question answering (span_extraction)
        This Module is composed of the BERT model with a linear on top of 
        the sequence output that compute start_logits, and end_logits. 
    Params:
        config: a BertConfig class instance with the configuration to build a new model. 
    Inputs:
        input_ids: torch.LongTensor. of shape [batch_size, sequence_length]
        token_type_ids: an optional torch.LongTensor, [batch_size, sequence_length]
            of the token type [0, 1]. Type 0 corresponds to sentence A, Type 1 corresponds to sentence B. 
        attention_mask: an optional torch.LongTensor of shape [batch_size, sequence_length]
            with index select [0, 1]. it is a mask to be used if the input sequence length is smaller 
            than the max input sequence length in the current batch. 
        start_positions: positions of the first token for the labeled span. torch.LongTensor 
            of shape [batch_size, seq_len], if current position is start of entity, the value equals to 1. 
            else the value equals to 0. 
        end_position: position to the last token for the labeled span. 
            torch.LongTensor, [batch_size, seq_len]
    Outputs:
        if "start_positions" and "end_positions" are not None
            output the total_loss which is the sum of the CrossEntropy loss 
            for the start and end token positions. 
        if "start_positon" or "end_positions" is None 
    """
    def __init__(self, config):
        super(BertMRCNER, self).__init__()
        bert_config = BertConfig.from_dict(config.bert_config.to_dict()) 
        self.bert = BertModel(bert_config)

        self.start_outputs = nn.Linear(config.hidden_size, 2) 
        self.end_outputs = nn.Linear(config.hidden_size, 2)         

        self.hidden_size = config.hidden_size 
        self.bert = self.bert.from_pretrained(config.bert_model) 

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
        start_positions=None, end_positions=None):

        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = sequence_output.view(-1, self.hidden_size)  
        
        start_logits = self.start_outputs(sequence_output) 
        end_logits = self.end_outputs(sequence_output) 


        if start_positions is not None and end_positions is not None:
            loss_fct = CrossEntropyLoss() 
        
            start_loss = loss_fct(start_logits.view(-1, 2), start_positions.view(-1))
            end_loss = loss_fct(end_logits.view(-1, 2), end_positions.view(-1))
            total_loss = start_loss + end_loss + span_loss
            # total_loss = (start_loss + end_loss) / 2 
            return total_loss 
        else:
            return start_logits, end_logits 
