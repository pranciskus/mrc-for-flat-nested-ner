#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.03.29 
# First create: 2019.02.13 
# Description:
# query_ner_data_utils.py 


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import csv 
import json
import random 
import logging 
import argparse 
import numpy as np 
from tqdm import tqdm 



import torch 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler 


from utils.bert_tokenizer import whitespace_tokenize 
from utils.bert_tokenizer import BertTokenizer4Tagger



class InputExample(object):
    def __init__(self, 
        qas_id, 
        question_text, 
        doc_text, 
        doc_tokens = None, 
        orig_answer_text=None, 
        start_position=None, 
        end_position=None, 
        is_impossible=None, 
        ner_cate=None):

        """
        Desc:
            is_impossible: bool, [True, False]
        """

        self.qas_id = qas_id 
        self.question_text = question_text 
        self.doc_text = doc_text 
        self.doc_tokens = doc_tokens 
        self.orig_answer_text = orig_answer_text 
        self.start_position = start_position 
        self.end_position = end_position 
        self.is_impossible = is_impossible 
        self.ner_cate = ner_cate 



class InputFeatures(object):
    """
    Desc:
        a single set of features of data 
    Args:
        start_pos: start position is a list of symbol 
        end_pos: end position is a list of symbol 
    """
    def __init__(self, 
        unique_id, 
        tokens,  
        input_ids, 
        input_mask, 
        segment_ids, 
        ner_cate, 
        start_position=None, 
        end_position=None, 
        is_impossible=None):

        self.unique_id = unique_id 
        self.tokens = tokens 
        self.input_mask = input_mask
        self.input_ids = input_ids 
        self.ner_cate = ner_cate 
        self.segment_ids = segment_ids 
        self.start_position = start_position 
        self.end_position = end_position 
        self.is_impossible = is_impossible 


def convert_examples_to_features(examples, tokenizer, label_lst, max_seq_length, is_training=True, 
    allow_impossible=True):
    label_map = {tmp: idx for idx, tmp in enumerate(label_lst)}
    features = []

    for (example_idx, example) in enumerate(examples):
        if not allow_impossible:
            if not example.is_impossible:
                continue 

        query_tokens = tokenizer.tokenize(example.question_text)

        whitespace_doc = whitespace_tokenize(example.doc_text)

        if example.start_position == -1 and example.end_position == -1:
            doc_start_pos = []
            doc_end_pos = []
            all_doc_tokens = []

            for token_item in whitespace_doc:
                tmp_subword_lst = tokenizer.tokenize(token_item)
                all_doc_tokens.extend(tmp_subword_lst)
            doc_start_pos = [0] * len(all_doc_tokens)
            doc_end_pos = [0] * len(all_doc_tokens)
        else:
            doc_start_pos = []
            doc_end_pos = []
            all_doc_tokens = []

            fake_start_pos = [0] * len(whitespace_doc)
            fake_end_pos = [0] * len(whitespace_doc)

            for start_item in example.start_position:
                fake_start_pos[start_item] = 1 
            for end_item in example.end_position:
                fake_end_pos[end_item] = 1 

            for idx, (token, start_label, end_label) in enumerate(zip(whitespace_doc, fake_start_pos, fake_end_pos)):
                tmp_subword_lst = tokenizer.tokenize(token)
                if len(tmp_subword_lst) > 1:
                    if start_label == 0: 
                        doc_start_pos.extend([0] * len(tmp_subword_lst)) 
                    if end_label == 0:
                        doc_end_pos.extend([0] * len(tmp_subword_lst))
                    if start_label != 0:
                        doc_start_pos.append(1)
                        doc_start_pos.extend([0] * (len(tmp_subword_lst) - 1))
                    if end_label != 0:
                        doc_end_pos.extend([0] * (len(tmp_subword_lst) - 1))
                        doc_end_pos.append(1)
                    all_doc_tokens.extend(tmp_subword_lst)
                elif len(tmp_subword_lst) == 1:
                    doc_start_pos.append(start_label)
                    doc_end_pos.append(end_label)
                    all_doc_tokens.extend(tmp_subword_lst) 
                else:
                    raise ValueError("Please check the result of tokenizer !!! !!! ")
  
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3 
       
        # print(all_doc_tokens)
        # print(doc_start_pos) 
        # print(len(all_doc_tokens))
        # print(len(doc_start_pos)) 
        assert len(all_doc_tokens) == len(doc_start_pos) 
        assert len(all_doc_tokens) == len(doc_end_pos) 
        assert len(doc_start_pos) == len(doc_end_pos) 

        if len(all_doc_tokens) >= max_tokens_for_doc:
            all_doc_tokens = all_doc_tokens[: max_tokens_for_doc]
            doc_start_pos = doc_start_pos[: max_tokens_for_doc]
            doc_end_pos = doc_end_pos[: max_tokens_for_doc]

        input_tokens = []
        segment_ids = []
        input_mask = []
        start_pos = []
        end_pos = []

        input_tokens.append("[CLS]")
        segment_ids.append(1) 
        input_mask.append(1) 
        start_pos.append(0) 
        end_pos.append(0)

        for query_item in query_tokens:
            input_tokens.append(query_item)
            segment_ids.append(1) 
            input_mask.append(1) 
            start_pos.append(0)
            end_pos.append(0)

        input_tokens.append("[SEP]")
        segment_ids.append(1) 
        input_mask.append(1) 
        start_pos.append(0) 
        end_pos.append(0) 


        input_tokens.extend(all_doc_tokens) 
        segment_ids.extend([1]* len(all_doc_tokens))
        input_mask.extend([1]* len(all_doc_tokens)) 
        start_pos.extend(doc_start_pos)
        end_pos.extend(doc_end_pos) 

        input_tokens.append("[SEP]")
        segment_ids.append(1)
        input_mask.append(1)
        start_pos.append(0)
        end_pos.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

        # zero-padding up to the sequence length 
        padding = [0] * (max_seq_length - len(input_ids)) 
        input_ids += padding 
        input_mask += padding 
        segment_ids += padding 
        start_pos += padding 
        end_pos += padding 
       

        features.append(
            InputFeatures(
                unique_id=example.qas_id, 
                tokens=input_tokens, 
                input_ids=input_ids, 
                input_mask=input_mask, 
                segment_ids=segment_ids, 
                start_position=start_pos, 
                end_position=end_pos, 
                is_impossible=example.is_impossible, 
                ner_cate=label_map[example.ner_cate]
                ))

    return features 



def read_query_ner_examples(input_file, is_training=True, with_negative=True):
    """
    Desc:
        read query-based NER data
    """
    input_data = []

    with open(input_file, "r") as f:
        data_lines = f.readlines()
        for data_item in data_lines:#[:len(data_lines)]: # todo
            data_item = data_item.strip()
            data_item = json.loads(data_item)
            input_data.append(data_item)


    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True 
        return False 

    examples = []
    for entry in input_data:
        qas_id = entry["qas_id"]
        question_text = entry["query"]
        doc_text = entry["context"]
        start_position = entry["start_position"]
        end_position = entry["end_position"]
        is_impossible = entry["impossible"]
        ner_cate = entry["ner_cate"]

        example = InputExample(qas_id=qas_id, 
            question_text=question_text, 
            doc_text=doc_text,
            start_position=start_position, 
            end_position=end_position, 
            is_impossible=is_impossible, 
            ner_cate=ner_cate)
        examples.append(example)
    print(len(examples))
    return examples  


