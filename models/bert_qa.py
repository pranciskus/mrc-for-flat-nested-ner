#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.04.28 
# FIrst create: 2019.04.28 
# Descripiton:
# bert model for Qestion Answering Task 


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



from pytorch_pretrained_bert.modeling import BertModel, PreTrainedBertModel 


class BertQA(PreTrainedBertModel):
    """
    Desc:
        BERT Model for question answering (span_extraction)
        This module is composed of the BERT mdoel with a linear on top of 
        the sequecne output that compute start_logits and end_logits. 
    Params:
        config: a BertConfig class instance with the configuration to build a new model. 
    Inputs:
        input_ids: torch.LongTensor, of shape [batch_size, sequence_lenght]
        token_type_ids: an optional torch.LongTensor, [batch_size, sequelcne_length]
            of the token type [0, 1]. Type 0 corresponds to "sentence A", Type 1 corresponds to "sentence B". 
        attention_mask: an optional torch.LongTensor of shape [batch_size, sequence_length], 
            with index selects [0, 1]. It is a mask to be used if the input sequnce length 
            is smaller than the max input sequence length in the current batch.   
        start_positions: positions fo the first token for the labeled span. torch.LongTensor
            of shape [batch_size], positions are clamped to the length of the sequence and position outside of 
            the sequence are not take into account for computing the loss. 
        end_position: position fo the last token for the labeled span. 
            torch.LongTensor, [batch_size], 
    Outputs:
        if "start_positions" and "end_positions" are not None
            output the total_loss which is the sum of the CrossEnropy loss 
            for the start and end token positions. 
        if "start_positions" or "end_postions" is None"
            output a tuple of start_logits, end_logits, which are the logits respectively 
            for the start and end position tokens of shape [batch_size, sequence_length]

    ########################################################################
    Examples usege:
        python
        # already been convered into WordPiece token ids:
        input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        config = BertConfig(vocab_size_or_config_json_file=32000, 
        hidden_size=768, num_hidden_layers=12, num_attention_heads=12, 
        intermediate_size=3072,)
    """
    def __init__(self, config):
        super(BertQA, self).__init__(config)
        self.bert = BertModel(config)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
        start_positions=None, end_positions=None):

        sequence_output, _ = self.bert(input_ids, token_type_ids, 
            attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # if we are on mulit-GPU, split add a dimension 
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometime the stat/ end positions are outsize our model inputs. 
            # we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2 
            return total_loss 
        else:
            return start_logits, end_logits 
