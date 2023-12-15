import datetime
import numpy as np
import os
import pickle
import sys
import time
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append('/home/ubuntu/workspace1/rmj/ECPE')

from ecpe_extractor.artifacts.config import *
from ecpe_extractor.artifacts.utils import *
from ecpe_extractor.data.data_loader import *
from ecpe_extractor.data.process_data import *
from ecpe_extractor.data.preprocess_tokenized_data import *
from ecpe_extractor.models.mvshif import MVSHIF

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # single GPU


def main(configs, fold_id, tokenizer):
    print(f"Starting training for Fold {fold_id}")

    # Load data
    data_path = 'ecpe_extractor/data/CV/fold{}'.format(fold_id) + '.pt'
    # data_path = 'ecpe_extractor/dataEnglish/CV/fold{}'.format(fold_id) + '.pt'
    total_data = torch.load(data_path)
    train_loader = build_dataset(configs, total_data['train'], mode='train')
    test_loader = build_dataset(configs, total_data['test'], mode='test')

    # Initialize model and optimizer
    model = MVSHIF(configs).to(DEVICE)
    params = list(model.named_parameters())
    optimizer_grouped_params = [
        {'params': [p for n, p in params if '_bert' in n], 'weight_decay': 0.01},
        {'params': [p for n, p in params if '_bert' not in n], 'lr': configs.lr, 'weight_decay': 0.01}
    ]
    optimizer = AdamW(params=optimizer_grouped_params, lr=configs.tuning_bert_rate)

    # scheduler
    training_steps = configs.epochs * len(train_loader) // configs.gradient_accumulation_steps
    warmup_steps = int(training_steps * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=training_steps)

    model.zero_grad()
    max_result_pair, max_result_emo, max_result_cau = None, None, None
    max_result_emos, max_result_caus = None, None
    early_stop_flag = None

    # Training loop
    for epoch in range(1, configs.epochs + 1):
        for train_step, batch in enumerate(train_loader, 1):
            model.train()

            _, clause_list, pairs, \
                feq, feq_mask, feq_seg, feq_len, fe_clause_len, fe_doc_len, fe_adj, feq_an, fe_an_mask, \
                fcq, fcq_mask, fcq_seg, fcq_len, fc_clause_len, fc_doc_len, fc_adj, fcq_an, fc_an_mask, \
                bcq, bcq_mask, bcq_seg, bcq_len, bc_clause_len, bc_doc_len, bc_adj, bcq_an, bc_an_mask, \
                beq, beq_mask, beq_seg, beq_len, be_clause_len, be_doc_len, be_adj, beq_an, be_an_mask \
                = batch

            clause_list, pairs, \
                feq, feq_mask, feq_seg, feq_len, fe_clause_len, fe_doc_len, fe_adj, feq_an, fe_an_mask, \
                fcq, fcq_mask, fcq_seg, fcq_len, fc_clause_len, fc_doc_len, fc_adj, fcq_an, fc_an_mask, \
                bcq, bcq_mask, bcq_seg, bcq_len, bc_clause_len, bc_doc_len, bc_adj, bcq_an, bc_an_mask, \
                beq, beq_mask, beq_seg, beq_len, be_clause_len, be_doc_len, be_adj, beq_an, be_an_mask \
                = clause_list, pairs, \
                feq.to(DEVICE), feq_mask.to(DEVICE), feq_seg.to(
                DEVICE), feq_len, fe_clause_len, fe_doc_len, fe_adj, feq_an, fe_an_mask, \
                fcq.to(DEVICE), fcq_mask.to(DEVICE), fcq_seg.to(
                DEVICE), fcq_len, fc_clause_len, fc_doc_len, fc_adj, fcq_an, fc_an_mask, \
                bcq.to(DEVICE), bcq_mask.to(DEVICE), bcq_seg.to(
                DEVICE), bcq_len, bc_clause_len, bc_doc_len, bc_adj, bcq_an, bc_an_mask, \
                beq.to(DEVICE), beq_mask.to(DEVICE), beq_seg.to(
                DEVICE), beq_len, be_clause_len, be_doc_len, be_adj, beq_an, be_an_mask

            f_emo_pred = model(feq, feq_mask, feq_seg, feq_len, fe_clause_len, fe_doc_len, fe_adj)
            f_cau_pred = model(fcq, fcq_mask, fcq_seg, fcq_len, fc_clause_len, fc_doc_len, fc_adj)
            b_emo_pred = model(beq, beq_mask, beq_seg, beq_len, be_clause_len, be_doc_len, be_adj)

            loss_e = model.loss_pre(f_emo_pred, feq_an, fe_an_mask)
            loss_ec = model.loss_pre(f_cau_pred, fcq_an, fc_an_mask)
            loss_ce = model.loss_pre(b_emo_pred, beq_an, be_an_mask)
            losses = (loss_e + loss_ec + loss_ce) / configs.gradient_accumulation_steps
            losses.backward()

            if train_step % configs.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if train_step % 200 == 0:
                print(f'Epoch: {epoch}, step: {train_step}, loss: {loss_e}, {loss_ec}, {loss_ce}')

        # Load emotion-related dictionary
        with open('/home/ubuntu/workspace1/rmj/ECPE/ecpe_extractor/data/sentimental_clauses.pkl', 'rb') as f:
            emo_dictionary = pickle.load(f)

        with torch.no_grad():
            eval_emo, eval_cau, eval_pair, eval_emos, eval_cuas = evaluate(test_loader, model, tokenizer,
                                                                           emo_dictionary)

            if max_result_pair is None or eval_pair[0] > max_result_pair[0]:
                early_stop_flag = 1
                max_result_emo = eval_emo
                max_result_cau = eval_cau
                max_result_pair = eval_pair

                state_dict = {'model': model.state_dict(), 'result': max_result_pair}
                torch.save(state_dict, 'ecpe_extractor/artifacts/model_fold{}.pth'.format(fold_id))
            else:
                early_stop_flag += 1
        if epoch > configs.epochs / 2 and early_stop_flag >= 7:
            break

    return max_result_emo, max_result_cau, max_result_pair, max_result_emos, max_result_caus


if __name__ == '__main__':
    start_time = time.time()
    configs = Config()
    tokenizer = AutoTokenizer.from_pretrained(configs.bert_cache_path)
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    result_e, result_c, result_p = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    res, rcs = [0, 0, 0], [0, 0, 0]

    processData()
    preprocess_dual_data()

    for fold_id in range(1, 11):

        print(f'Fold {fold_id} - Begin Training')

        metric_e, metric_c, metric_pair, es, cs = main(configs, fold_id, tokenizer)
        print(f'Best ECPE - Precision: {metric_pair[1]}, Recall: {metric_pair[2]}, F1: {metric_pair[0]}')
        print(f'Best ECE  - Precision: {metric_e[1]}, Recall: {metric_e[2]}, F1: {metric_e[0]}')
        print(f'Best CCE  - Precision: {metric_c[1]}, Recall: {metric_c[2]}, F1: {metric_c[0]}')

        for i in range(3):
            result_e[i] += metric_e[i]
            result_c[i] += metric_c[i]
            result_p[i] += metric_pair[i]

        print(f'Fold {fold_id} - Training Completed')

    for i in range(3):
        result_e[i] /= 10
        result_c[i] /= 10
        result_p[i] /= 10

    print(f'ECPE - Precision: {result_p[1]}, Recall: {result_p[2]}, F1: {result_p[0]}')
    print(f'ECE  - Precision: {result_e[1]}, Recall: {result_e[2]}, F1: {result_e[0]}')
    print(f'CCE  - Precision: {result_c[1]}, Recall: {result_c[2]}, F1: {result_c[0]}')

    print("Total Time:{}".format(time.time() - start_time))
