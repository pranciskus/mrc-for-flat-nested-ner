#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.03.23 
# First create: 2019.03.23 
# Description:
# run_machine_comprehension.py 
# Please Notice that the data should contain 
# multi answers 
# need pay MORE attention when loading data 



import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import csv 
import json 
import argparse 
import numpy as np 
from tqdm import tqdm 


import torch 
from torch import nn 
from torch.optim import Adam 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
SequentialSampler 
from torch.utils.data.distributed import DistributedSampler 


from pytorch_pretrained_bert.tokenization import BertTokenizer  
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear 
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE 


from data.model_config import Config 
from data.query_ner_data_processor import * 
from data.query_ner_data_utils import convert_examples_to_features
from models.bert_mrc_ner import BertQueryNER
from layers.math_funcs import sigmoid2label
from utils.bert_tokenizer import BertTokenizer4Tagger 
from utils.query_ner_evaluate  import query_ner_compute_performance


def args_parser():
    # start parser 
    parser = argparse.ArgumentParser()

    # requires parameters 
    parser.add_argument("--config_path", default="/home/lixiaoya/", type=str)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--bert_model", default=None, type=str,)
    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--dev_batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--checkpoint", default=100, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3006)
    parser.add_argument("--export_model", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="/home/lixiaoya/output")
    parser.add_argument("--data_sign", type=str, default="msra_ner")

    args = parser.parse_args()

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps 

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed_all(args.seed)

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError 
    os.makedirs(args.output_dir, exist_ok=True)

    return args



def load_data(config):

    print("-*-"*10)
    print("current data_sign: {}".format(config.data_sign))

    if config.data_sign == "conll03":
        data_processor = Conll03Processor()
    elif config.data_sign == "msra":
        data_processor = MSRAProcessor()
    elif config.data_sign == "zh_onto":
        data_processor = OntoZhProcessor()
    elif config.data_sign == "en_onto":
        data_processor = OntoEngProcessor()
    elif config.data_sign == "genia":
        data_processor = GeniaProcessor()
    elif config.data_sign == "ace2004":
        data_processor = ACE2004Processor()
    elif config.data_sign == "ace2005":
        data_processor = ACE2005Processor()
    elif config.data_sign == "kbp17":
        data_processor =KBP17Processor()
    else:
        raise ValueError("Please Notice that your data_sign DO NOT exits !!!!!")

    label_list = data_processor.get_labels()
    tokenizer = BertTokenizer4Tagger.from_pretrained(config.bert_model, do_lower_case=True)

    # load data exampels 
    train_examples = data_processor.get_train_examples(config.data_dir)
    dev_examples = data_processor.get_dev_examples(config.data_dir)
    test_examples = data_processor.get_test_examples(config.data_dir)

    # convert data example into featrues
    train_features = convert_examples_to_features(train_examples, tokenizer, label_list, config.max_seq_length, allow_impossible=False) 
    print("check loaded data")
    print(train_features[0])
    train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    train_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    train_ner_cate = torch.tensor([f.ner_cate for f in train_features], dtype=torch.long)
    train_start_pos = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    train_end_pos = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    # train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_ids)
    train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_start_pos, train_end_pos, train_ner_cate)
    # train_sampler = DistributedSampler(train_data)
    train_sampler = RandomSampler(train_data)


    dev_features = convert_examples_to_features(dev_examples, tokenizer, label_list, config.max_seq_length, allow_impossible=False)
    dev_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    dev_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
    dev_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
    dev_start_pos = torch.tensor([f.start_position for f in dev_features], dtype=torch.long)
    dev_end_pos = torch.tensor([f.end_position for f in dev_features], dtype=torch.long)
    dev_ner_cate = torch.tensor([f.ner_cate for f in dev_features], dtype=torch.long)
    dev_data = TensorDataset(dev_input_ids, dev_input_mask, dev_segment_ids, dev_start_pos, dev_end_pos, dev_ner_cate)
    dev_sampler = RandomSampler(dev_data)

    
    test_features = convert_examples_to_features(test_examples, tokenizer, label_list, config.max_seq_length, allow_impossible=False)
    test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_start_pos = torch.tensor([f.start_position for f in test_features], dtype=torch.long)
    test_end_pos = torch.tensor([f.end_position for f in test_features], dtype=torch.long)
    test_ner_cate = torch.tensor([f.ner_cate for f in test_features], dtype=torch.long)
    test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_start_pos, test_end_pos, test_ner_cate)
    test_sampler = RandomSampler(test_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, \
        batch_size=config.train_batch_size)

    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, \
        batch_size=config.dev_batch_size)

    test_dataloader = DataLoader(test_data, sampler=test_sampler, \
        batch_size=config.test_batch_size)

    num_train_steps = int(len(train_examples) / config.train_batch_size * config.num_train_epochs) 
    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps, label_list 



def load_model(config, num_train_steps, label_list):
    # device = torch.device(torch.cuda.is_available())
    device = torch.device("cuda") 
    n_gpu = torch.cuda.device_count()
    model = BertQueryNER(config, ) 
    # model = BertForTagger.from_pretrained(config.bert_model, num_labels=13)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # prepare  optimzier 
    param_optimizer = list(model.named_parameters())

        
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
    {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
    {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    # optimizer = Adam(optimizer_grouped_parameters, lr=config.learning_rate) 
    optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=config.warmup_proportion, t_total=num_train_steps, max_grad_norm=config.clip_grad) 

    return model, optimizer, device, n_gpu



def train(model, optimizer, train_dataloader, dev_dataloader, test_dataloader, config, \
    device, n_gpu, label_list):
    global_step = 0 
    nb_tr_steps = 0 
    tr_loss = 0 

    dev_best_acc = 0 
    dev_best_precision = 0 
    dev_best_recall = 0 
    dev_best_f1 = 0 
    dev_best_loss = 10000000000000


    test_best_acc = 0 
    test_best_precision = 0 
    test_best_recall = 0 
    test_best_f1 = 0 
    test_best_loss = 1000000000000000

    model.train()

    for idx in range(int(config.num_train_epochs)):
        tr_loss = 0 
        nb_tr_examples, nb_tr_steps = 0, 0 
        print("#######"*10)
        print("EPOCH: ", str(idx))
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch) 
            input_ids, input_mask, segment_ids, start_pos, end_pos, ner_cate = batch 
            loss = model(input_ids, segment_ids, input_mask, start_pos, end_pos)
            if n_gpu > 1:
                loss = loss.mean()

            model.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad) 

            tr_loss += loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1 

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                # optimizer.zero_grad()
                global_step += 1 

            if nb_tr_steps % config.checkpoint == 0:
                print("-*-"*15)
                print("current training loss is : ")
                print(loss.item())
                # continue 
                tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1 = eval_checkpoint(model, dev_dataloader, config, device, n_gpu, label_list, eval_sign="dev")
                print("......"*10)
                print("DEV: loss, acc, precision, recall, f1")
                print(tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1)

                if tmp_dev_f1 > dev_best_f1 or tmp_dev_acc > dev_best_acc:
                    dev_best_acc = tmp_dev_acc 
                    dev_best_loss = tmp_dev_loss 
                    dev_best_precision = tmp_dev_prec 
                    dev_best_recall = tmp_dev_rec 
                    dev_best_f1 = tmp_dev_f1 

                    tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1 = eval_checkpoint(model, test_dataloader, config, device, n_gpu, label_list, eval_sign="test")
                    print("......"*10)
                    print("TEST: loss, acc, precision, recall, f1")
                    print(tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1)

                    if tmp_test_f1 > test_best_f1 or tmp_test_acc > test_best_acc:
                        test_best_acc = tmp_test_acc 
                        test_best_loss = tmp_test_loss 
                        test_best_precision = tmp_test_prec 
                        test_best_recall = tmp_test_rec 
                        test_best_f1 = tmp_test_f1 

                        # export model 
                        if config.export_model:
                            model_to_save = model.module if hasattr(model, "module") else model 
                            output_model_file = os.path.join(config.output_dir, "bert_finetune_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)

                print("-*-"*15)

    # export a trained mdoel 
    model_to_save = model 
    output_model_file = os.path.join(config.output_dir, "bert_model.bin")
    if config.export_model == "True":
        torch.save(model_to_save.state_dict(), output_model_file)


    print("=&="*15)
    print("DEV: current best precision, recall, f1, acc, loss ")
    print(dev_best_precision, dev_best_recall, dev_best_f1, dev_best_acc, dev_best_loss)
    print("TEST: current best precision, recall, f1, acc, loss ")
    print(test_best_precision, test_best_recall, test_best_f1, test_best_acc, test_best_loss)
    print("=&="*15)



def eval_checkpoint(model_object, eval_dataloader, config, \
    device, n_gpu, label_list, eval_sign="dev"):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader 
    model_object.eval()

    idx2label = {i: label for i, label in enumerate(label_list)}

    eval_loss = 0 
    start_pred_lst = []
    end_pred_lst = []
    mask_lst = []
    start_gold_lst = []
    end_gold_lst = []
    eval_steps = 0 
    ner_cate_lst = [] 

    for input_ids, input_mask, segment_ids, start_pos, end_pos, ner_cate in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_pos = start_pos.to(device)
        end_pos = end_pos.to(device)

        with torch.no_grad():
            tmp_eval_loss = model_object(input_ids, segment_ids, input_mask, start_pos, end_pos)
            start_logits, end_logits = model_object(input_ids, segment_ids, input_mask)

        start_logits = start_logits.detach().cpu().numpy()
        end_logits = end_logits.detach().cpu().numpy()
        # logits = np.argmax(logits, axis=-1)
        start_pos = start_pos.to("cpu").numpy()
        end_pos = end_pos.to("cpu").numpy()
        input_mask = input_mask.to("cpu").numpy().tolist()
        reshape_lst = start_pos.shape
        ner_cate = ner_cate.numpy() 
        # -----------------------------------------------------------------------------------------------------------
        # start logits, end logits 
        # ------------------------------------------------------------------------------------------------------------
        start_logits = np.reshape(start_logits, (reshape_lst[0], reshape_lst[1], 2)).tolist()
        end_logits = np.reshape(end_logits, (reshape_lst[0], reshape_lst[1], 2)).tolist()

        start_label = np.argmax(start_logits, axis=-1).tolist()  
        end_label = np.argmax(end_logits, axis=-1).tolist()
        # start_label = sigmoid2label(start_logits)
        # end_label = sigmoid2label(end_logits)

        start_pos = start_pos.tolist()
        end_pos = end_pos.tolist()
        
        ner_cate = ner_cate.tolist() 
        eval_loss += tmp_eval_loss.mean().item()

        start_pred_lst += start_label 
        end_pred_lst += end_label 
        start_gold_lst += start_pos 
        end_gold_lst += end_pos 
        ner_cate_lst += ner_cate 

        mask_lst += input_mask 
        eval_steps += 1   
    

    eval_accuracy, eval_precision, eval_recall, eval_f1 = query_ner_compute_performance(start_pred_lst, end_pred_lst, start_gold_lst, end_gold_lst, ner_cate_lst, label_list, mask_lst, dims=2)  
    # eval_accuracy, eval_precision, eval_recall, eval_f1 = compute_performance(pred_lst, gold_lst, mask_lst, label_list, dims=2)  

    average_loss = round(eval_loss / eval_steps, 4)  
    eval_f1 = round(eval_f1 , 4)
    eval_precision = round(eval_precision , 4)
    eval_recall = round(eval_recall , 4) 
    eval_accuracy = round(eval_accuracy , 4) 

    return average_loss, eval_accuracy, eval_precision, eval_recall, eval_f1 



def merge_config(args_config):
    model_config_path = args_config.config_path 
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    model_config.print_config()
    return model_config



def main():
    args_config = args_parser()
    config = merge_config(args_config)
    train_loader, dev_loader, test_loader, num_train_steps, label_list = load_data(config)
    model, optimizer, device, n_gpu = load_model(config, num_train_steps, label_list)
    train(model, optimizer, train_loader, dev_loader, test_loader, config, device, n_gpu, label_list)
    

if __name__ == "__main__":
    main() 
