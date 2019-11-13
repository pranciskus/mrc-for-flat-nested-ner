#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


# Author: Xiaoy LI
# Last update: 2019.04.29 
# First create: 2019.03.23 
# Description:
# preprocess_data.py
# -----------------------------------------------------------------
# Load data in CoNLL 2003 format 
# Dump data to query_based NER format 
# 1. every sentence should have multi-question
# -----------------------------------------------------------------
# data produce:
# 1. query, 2.context, 3. start_pos, 4. end_pos, 5. impossible 6. qas_id 


import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import json

from data.data_generate.file_utils import load_conll
from data.data_generate.query_map import query_sign_map
from data.data_generate.tagging_utils import get_span_labels


def gen_query_ner(dataset, query_info_dict, query_type="psedo_query"):
    query_ner_dataset = []

    dataset_label_lst = query_info_dict["tags"]
    print("Check label list")
    print(dataset_label_lst)
    print("-*-" * 10)
    tmp_qas_id = 0
    for idx, (word_lst, label_lst) in enumerate(dataset):
        candidate_span_label = get_span_labels(label_lst)
        for label_idx, tmp_label in enumerate(dataset_label_lst):
            tmp_qas_id += 1
            tmp_query = query_info_dict[query_type][tmp_label]
            # tmp_query = " ".join(list(tmp_query))
            tmp_context = " ".join(word_lst)
            tmp_start_pos = []
            tmp_end_pos = []

            start_end_label = [(start, end) for start, end, label_content in candidate_span_label if
                               label_content == tmp_label]
            if len(start_end_label) != 0:
                for span_item in start_end_label:
                    start_idx, end_idx = span_item
                    tmp_start_pos.append(start_idx)
                    tmp_end_pos.append(end_idx)
                tmp_possible = True
            else:
                tmp_possible = False
                tmp_start_pos = -1
                tmp_end_pos = -1
            query_ner_dataset.append({
                "qas_id": tmp_qas_id,
                "query": tmp_query,
                "context": tmp_context,
                "ner_cate": tmp_label,
                "start_position": tmp_start_pos,
                "end_position": tmp_end_pos,
                "impossible": tmp_possible,
            })

    return query_ner_dataset


def main(source_data_repo, source_file_name, target_data_repo, target_file_name, dataset_sign=None,
         query_type="psedo_query"):
    file_path = os.path.join(source_data_repo, source_file_name)
    dataset = load_conll(file_path)
    # [([word_lst], [label_list]), ([word_lst,], (label_lst))]
    query_info_dict = query_sign_map[dataset_sign]
    # query_info_dict["tags"] = ["ORG", "PER", "LOC"]
    # query_info_dict["psedo_query"]
    # query_info_dict["psedo_query"]["ORG"] = "people name"
    query_ner_dataset = gen_query_ner(dataset, query_info_dict, query_type=query_type)

    print("-*-" * 10)
    print("check the content of generated data")
    print(query_ner_dataset[0])
    target_file_path = os.path.join(target_data_repo, target_file_name)
    with open(target_file_path, "w") as f:
        for target_data_item in query_ner_dataset:
            target_data_item = json.dumps(target_data_item, ensure_ascii=False) + "\n"
            f.write(target_data_item)


if __name__ == "__main__":
    import os
    data_repo = "/data/nfsdata/data/yuxian/datasets/genia"

    for prefix in ["train", "dev", "test"]:
        source_file_name = f"{prefix}.ner"
        target_file_name = f"query_ner.{prefix}"

        target_data_repo = "/data/data_repo/genia_ner"
        os.makedirs(target_data_repo, exist_ok=True)
        dataset_sign = "genia_ner"
        main(data_repo, source_file_name, target_data_repo, target_file_name, dataset_sign=dataset_sign,
             query_type="natural_query")
