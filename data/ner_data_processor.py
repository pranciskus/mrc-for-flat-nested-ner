#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.04.30 
# First create: 2019.04.30 
# Description:
# query_ner_data_processor.py


import os
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import csv 
import random 
import logging 
import argparse 
import numpy as np 
from tqdm import tqdm 


import torch 
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler 


from data.query_ner_data_utils import read_query_ner_examples 



class QueryNERProcessor(object):
    # processor for the query-based ner dataset 
    def get_train_examples(self, data_dir):
        data = read_query_ner_examples(os.path.join(data_dir, "query_ner.train"))
        return data#[:len(data)//32]

    def get_dev_examples(self, data_dir):
        return read_query_ner_examples(os.path.join(data_dir, "query_ner.dev"))

    def get_test_examples(self, data_dir):
        return read_query_ner_examples(os.path.join(data_dir, "query_ner.test"))


class Conll03Processor(QueryNERProcessor):
    def get_labels(self, ):
        return ["ORG", "PER", "LOC", "MISC", "O"]


class MSRAProcessor(QueryNERProcessor):
    def get_labels(self, ):
        return ["NS", "NR", "NT", "O"]


class OntoZhProcessor(QueryNERProcessor):
    def get_labels(self, ):
        return ["LOC", "PER", "GPE", "ORG", "O"]


class OntoEngProcessor(QueryNERProcessor):
    def get_labels(self, ):
        return ['ORDINAL', 'CARDINAL', 'LOC', 'WORK_OF_ART', 'LANGUAGE', 'ORG', 'FAC', 'PERSON', 'EVENT', 'TIME', 'LAW', 'NORP', 'PERCENT', 'DATE', 'GPE', 'QUANTITY', 'O', 'PRODUCT', 'MONEY']


class ResumeZhProcessor(QueryNERProcessor):
    def get_labels(self, ):
        return ["ORG", "LOC", "NAME", "RACE", "TITLE", "EDU", "PRO", "CONT", "O"]


class GeniaProcessor(QueryNERProcessor):
    def get_labels(self, ):
        return ['cell_line', 'cell_type', 'DNA', 'RNA', 'protein', "O"]
