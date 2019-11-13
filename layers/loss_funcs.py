#!/usr/bin/env python3
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.03.23 
# First create: 2019.03.23 
# Description:
# loss_funcs_examples.py



import os 
import sys 
import numpy as np 



root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)



import torch 
import torch.nn as nn 
from torch.nn import BCEWithLogitsLoss


def nll_loss():
    # input size if N x C = 3 x 5 
    input = torch.randn(3, 5, requires_grad=True)
    # each element in target has to have 0 <= value < C 
    target = torch.tensor([1, 0, 4])
    output = F.nll_loss(F.log_softmax(input), target)
    output.backward()



def cross_entropy_loss():
    # loss 
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    output.backward()


def bce_logits_loss():
    """
    Desc:
        Input: math. (N, *) where * means, any number of additional dimensions 
        Target: math, (N, *) where the same shape as the input. 
    """
    loss = nn.BCEWithLogitsLoss()
    input = torch.randn(3, requires_grad=True)
    target = torch.empty(3).random_(2)
    output = loss(input, target)
    output.backward()
