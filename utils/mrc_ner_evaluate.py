#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.02.12 
# First create: 2019.02.12 
# Description:
# 



import os 
import sys 
import math 
import numpy as np 
from copy import deepcopy

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


from utils.ner_evaluate import extract_entities
from data.data_generate.tagging_utils import get_tags, get_span_labels 


def update_label_lst(label_lst):
    """
    Desc:
        label_lst is a list of entity category such as: ["NS", "NT", "NM"]
        after update, ["B-NS", "E-NS", "S-NS"]
    """
    update_label_lst = [] 
    for label_item in label_lst:
        if label_item != "O":
            update_label_lst.append("B-{}".format(label_item))
            update_label_lst.append("E-{}".format(label_item))
            update_label_lst.append("S-{}".format(label_item)) 
        else:
            update_label_lst.append(label_item) 
    return update_label_lst 


def split_index(label_list):
    label_dict = {label: i for i, label in enumerate(label_list)}
    label_idx = [tmp_value for tmp_key, tmp_value in label_dict.items() if "S" in tmp_key.split("-")[0] or "B" in tmp_key]
    str_label_idx = [str(tmp) for tmp in label_idx]
    label_idx = "_".join(str_label_idx)
    label2idx = label_dict 
    return label_idx, label2idx  


def query_ner_compute_performance(pred_start, pred_end, 
    gold_start, gold_end, ner_cate, label_lst, mask_index, dims=2):
    
    cate_idx2label = {idx: value for idx, value in enumerate(label_lst)} 
    up_label_lst = update_label_lst(label_lst) 
    split_label_idx, label2idx = split_index(up_label_lst)
    idx2label = {v: k for k, v in label2idx.items()}
    start_label = split_label_idx
    # print(start_label) 
    if dims == 1:
        len_labels = len(gold_start)
        ner_cate = cate_idx2label[ner_cate]
        gold_start = [idx for idx, tmp in enumerate(gold_start) if tmp != 0]
        gold_end = [idx for idx, tmp in enumerate(gold_end) if tmp != 0]
        gold_label = ["O"] * len_labels 
        for start_item in gold_start:
            gold_label[start_item] = "B-{}".format(ner_cate)
        for end_item in gold_end:
            gold_label[end_item] = "E-{}".format(ner_cate)
        single_gold = list(set(gold_start) & set(gold_end)) 
        for single_item in single_gold:
            gold_label[single_item] = "S-{}".format(ner_cate) 

        # if len(gold_start) == 0 and len(gold_end) == 0:
        #     gold_label = len_labels * ["O"] 
        # else:
        #     gold_span = [(s_tmp, e_tmp, ner_cate) for s_tmp, e_tmp in zip(gold_start, gold_end)]
        #     gold_label = get_tags(gold_span, len_labels, "BIOES")

        pred_start = [idx for idx, tmp in enumerate(pred_start) if tmp != 0]
        pred_end = [idx for idx, tmp in enumerate(pred_end) if tmp !=0]
        pred_label = ["O"] * len_labels
        for start_item in pred_start:
            pred_label[start_item] = "B-{}".format(ner_cate)
        for end_item in pred_end:
            pred_label[end_item] = "E-{}".format(ner_cate)
        single_pred = list(set(pred_start) & set(pred_end)) 
        for single_item in single_pred:
            pred_label[single_item] = "S-{}".format(ner_cate) 

        pred_label = [label2idx[tmp] for tmp in pred_label]
        gold_label = [label2idx[tmp] for tmp in gold_label]

        mask_index = [tmp_idx for tmp_idx, tmp in enumerate(mask_index) if tmp != 0]
        pred_label = [tmp for tmp_idx, tmp in enumerate(pred_label) if tmp_idx in mask_index]
        gold_label = [tmp for tmp_idx, tmp in enumerate(gold_label) if tmp_idx in mask_index]

        pred_entities = extract_entities(pred_label, start_label = start_label)
        truth_entities = extract_entities(gold_label, start_label = start_label)

        # print("ped_entity") 
        # print(pred_entities)
        # print("truth_entiyt")
        # print(truth_entities) 
        # print("-*-"*10)
        num_true = len(truth_entities)
        num_extraction = len(pred_entities)

        num_true_positive = 0 
        for entity_idx in pred_entities.keys():
            try:
                if truth_entities[entity_idx] == pred_entities[entity_idx]:
                    num_true_positive += 1 
            except:
                pass 

        dict_match = list(filter(lambda x: x[0] == x[1], zip(pred_label, gold_label)))
        if len(gold_label) != 0:
            acc = len(dict_match) / float(len(gold_label))
        else:
            acc = 1

        return acc, num_true_positive, float(num_extraction), float(num_true)

    elif dims == 2:
        acc, posit, extra, true = 0, 0, 0, 0

        # pred_start, pred_end, gold_start, gold_end, ner_cate, label_lst, mask_index,
        # for pred_item, truth_item, mask_item in zip(pred_label, gold_label, pred_mask):
        for pred_start_item, pred_end_item, gold_start_item, gold_end_item, ner_cate_item, mask_index_item in zip(pred_start, 
            pred_end, gold_start, gold_end, ner_cate, mask_index): 
            tmp_acc, tmp_posit, tmp_extra, tmp_true = query_ner_compute_performance(pred_start_item, pred_end_item, gold_start_item, 
                gold_end_item, ner_cate_item, label_lst, mask_index_item, dims=1) 
            posit += tmp_posit 
            extra += tmp_extra 
            true += tmp_true 
            acc += tmp_acc 

        if extra != 0:
            pcs = posit / float(extra)
        else:
            pcs = 0 

        if true != 0:
            recall = posit / float(true)
        else:
            recall = 0 

        if pcs + recall != 0 :
            f1 = 2 * pcs * recall / (pcs + recall)
        else:
            f1 = 0 
        acc = acc / len(pred_start)
        acc, pcs, recall, f1 = round(acc, 4), round(pcs, 4), round(recall, 4), round(f1, 4)
        return acc, pcs, recall, f1 


if __name__ == "__main__":
    pass 
