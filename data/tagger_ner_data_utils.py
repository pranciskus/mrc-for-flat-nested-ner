#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.03.23 
# First create: 2019.03.23 
# Description:
# tagging_data_utils.py 


import os 
import sys
import csv 
import logging
import argparse 
import random 
import numpy as np 
from tqdm import tqdm, trange 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch 
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, 
SequentialSampler 


from data.apply_text_norm import process_sent 



class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid 
        self.text_a = text_a 
        self.text_b = text_b 
        self.label = label 



class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids 
        self.input_mask = input_mask 
        self.segment_ids = segment_ids 
        self.label_id = label_id 



class DataProcessor(object):
    # base class for data converts for sequences class datasets 
    def get_train_examples(self, data_dir):
        # get a collections of "InputExample" for train set 
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        # gets the collections of "InputExample" for dev set 
        raise NotImplementedError()

    def get_labels(self):
        # gets the list of labels for this dat aset 
        raise NotImplementedError()

    @classmethod 
    def _read_tsv(cls, input_file, quotechar=None):
        # read a tab seperated value file 
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines 


def convert_examples_to_features(examples, label_list, 
    max_seq_length, tokenizer, task_sign="ner"):
    
    # load a data file into a list of "InputBatch"
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        if len(tokens_s) > max_seq_length - 2:
            tokens_a = tokens_a[: (max_seq_length - 2)]

        tokens = ["[CLS]"] + toknes_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        # zero padding up to the sequence length 
        padding = [0]* (max_seq_length - len(input_ids))
        input_ids += padding 
        input_mask += padding 
        segment_ids += padding 

        assert len(input_ids) == max_seq_length 
        assert len(input_mask) == max_seq_length 
        assert len(segment_ids) == max_seq_length 


        if len(example.label) > max_seq_length - 2:
            example.label = example.label[: (max_seq_length - 2)]

        if task_sign == "ner":
            label_id = [label_map["O"]] + [label_map[tmp] for tmp in example.label] + [label_map["O"]]
            label_id += (len(input_ids) - len(label_id)) * [label_map["O"]]
        else:
            raise ValueError 

        features.append(InputFeature(
            input_ids=input_ids, input_mask=input_mask, 
            segment_ids=segment_ids, label_id=label_id))

    return features 