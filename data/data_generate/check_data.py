#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.04.29 
# FIrst create: 2019.04.29 
# Description:
# check and collect the data 
# -----------------------------------------------------------------------------------------------------------
# {'answers': [{'answer_start': 177, 'text': 'Denver Broncos'},
# {'answer_start': 177, 'text': 'Denver Broncos'}, 
# {'answer_start': 177, 'text': 'Denver Broncos'}], 
# 'question': 'Which NFL team represented the AFC at Super Bowl 50?', 
# 'id': '56be4db0acb8001400a502ec'}
# ------------------------------------------------------------------------------------------------------------


import os 
import sys 
import json


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
print("check root_path")
print(root_path)
if root_path not in sys.path: 
    sys.path.insert(0, root_path)


from data.query_ner_data_processor import * 
from utils.bert_tokenizer import BertTokenizer4Tagger 

def entity_per_sent():
    """
    Desc:
        num of entity per sentence 
    """
    pass 



def stat_entity_category(data_path):
    print("Please Notice that file data format should be CoNLL 2003 ~~~")
    category_lst = []
    with open(data_path, "r") as f:
        data_lines = f.readlines()
        for data_item in data_lines:
            if data_item == "\n":
                continue 
            data_item = data_item.strip()
            context_item, label_item = data_item.split(" ")
            category_lst.append(label_item.split("-")[-1])
        category_lst = list(set(category_lst))
        print("entity Category in data file is ")
        print(category_lst)
    return category_lst 


def length_stat(data_repo, data_sign, bert_model):
    def concate_sentence(input_example, tokenizer):
        sent = []
        sent = ["[CLS]"]

        query_sent = tokenizer.tokenize(input_example.question_text)
        sent.extend(query_sent)

        sent.append("[SEP]")

        doc_sent = tokenizer.tokenize(input_example.doc_text)
        sent.extend(doc_sent)

        sent.append("[SEP]")

        return sent 

    if data_sign == "conll03":
        data_processor = Conll03Processor()
    elif data_sign == "msra":
        data_processor = MSRAProcessor()
    elif data_sign == "zh_onto":
        data_processor = OntoZhProcessor()
    elif data_sign == "en_onto":
        data_processor = OntoEngProcessor()
    else:
        raise ValueError 

    label_list = data_processor.get_labels()
    tokenizer = BertTokenizer4Tagger.from_pretrained(bert_model, do_lower_case=True)

    train_examples = data_processor.get_train_examples(data_repo)
    dev_examples = data_processor.get_dev_examples(data_repo)
    test_examples = data_processor.get_test_examples(data_repo)

    train_len = [len(concate_sentence(tmp, tokenizer)) for tmp in train_examples]
    dev_len = [len(concate_sentence(tmp, tokenizer)) for tmp in dev_examples]
    test_len = [len(concate_sentence(tmp, tokenizer)) for tmp in test_examples]

    avg_train_len = round(sum(train_len) / len(train_len), 4)
    avg_dev_len = round(sum(dev_len) / len(dev_len), 4)
    avg_test_len = round(sum(test_len) / len(test_len), 4)

    max_train_len = max(train_len)
    max_dev_len = max(dev_len)
    max_test_len = max(test_len)

    threshold = 256
    one_train = [tmp for tmp in train_len if tmp > threshold]
    one_dev = [tmp for tmp in dev_len if tmp > threshold]
    one_test = [tmp for tmp in test_len if tmp > threshold]

    print("total num in train, dev, test")
    print(len(train_len), len(dev_len), len(test_len)) 
    print("max_len in train, dev, test")
    print(max_train_len, max_dev_len, max_test_len)
    print("average_len in train, dev, test")
    print(avg_train_len, avg_dev_len, avg_test_len)
    print(f"lens large than {threshold} in train, dev, test")
    print(len(one_train), len(one_dev), len(one_test))
    print("&=&"*10)


if __name__ == "__main__":
    data_repo = "zh_onto"
    data_sign = "zh_onto"
    bert_model = "/data/nfsdata/nlp/BERT_BASE_DIR/cased_L-24_H-1024_A-16"
    length_stat(data_repo, data_sign, bert_model) 
