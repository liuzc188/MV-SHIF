# @Time : 2023/11/19 20:28
# @Author : Cheng Yang
# @File ：run_experiment_v1.py

import json
import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import os
import random
import scipy.sparse as sp
import time
import torch
import torch.nn.functional as F
import warnings
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, BertConfig, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")

# ========================= 超参数设置 =========================
SEED = 126  # 37199, 129, 130, 144
epochs = 30
batch_size = 1
lr = 1e-5  # 1e-5
step1_threshold = 0.5  # 预测结果的概率阈值
step2_threshold = 0.3
step3_threshold = 0.3
chinese = False

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

if chinese:
    truncation_length = 300  # <= 512 chinese

    # BERT_PATH = '/home/ubuntu/workspace1/rmj/ECPE/ecpe_extractor/artifacts/bert_chinese'  # chinese
    BERT_PATH = '/home/ubuntu/workspace1/rmj/LJP/PLM/nezha'

    dataset_folder_path = '/home/ubuntu/workspace1/rmj/ECPE/ecpe_extractor/data/CV/'  # chinese dataset

    emotion_query_template = "这是情感子句吗?"
    all_cause_query_template = '这句话对应的原因子句有哪些?'
    one_emotion_query_template = '这句话对应的情感子句是哪一句?'
else:
    truncation_length = 370  # <= 512 english

    # BERT_PATH = '/home/ubuntu/workspace1/rmj/ECPE/ecpe_extractor/artifacts/bert'  # english
    BERT_PATH = '/home/ubuntu/workspace1/rmj/ECPE/ecpe_extractor/artifacts/roberta_english'  # english

    dataset_folder_path = '/home/ubuntu/workspace1/rmj/ECPE/ecpe_extractor/dataEnglish/CV/'  # english dataset

    emotion_query_template = "Is this an emotional clause?"
    all_cause_query_template = 'The corresponding causal clauses for this sentence are?'
    one_emotion_query_template = 'Which sentence is the emotional clause corresponding to this sentence?'


# ========================= 数据加载 =========================
class Train_Dataset(object):
    def __init__(self, dataset_step1, dataset_step2, dataset_step3, dataset_emotion_clause_idx_list,
                 dataset_emotion_cause_dic, dataset_cause_clause_idx_list, dataset_cause_emotion_dic):
        self.dataset_step1 = dataset_step1
        self.dataset_step2 = dataset_step2
        self.dataset_step3 = dataset_step3
        self.dataset_emotion_clause_idx_list = dataset_emotion_clause_idx_list
        self.dataset_emotion_cause_dic = dataset_emotion_cause_dic
        self.dataset_cause_clause_idx_list = dataset_cause_clause_idx_list
        self.dataset_cause_emotion_dic = dataset_cause_emotion_dic

    def __getitem__(self, index):
        return self.dataset_step1[index], self.dataset_step2[index], self.dataset_step3[index], \
            self.dataset_emotion_clause_idx_list[index], self.dataset_emotion_cause_dic[index], \
            self.dataset_cause_clause_idx_list[index], self.dataset_cause_emotion_dic[index]

    def __len__(self):
        return len(self.dataset_step1)


class Test_Dataset(object):
    def __init__(self, dataset_step1, dataset_emotion_clause_idx_list, dataset_emotion_cause_dic,
                 dataset_cause_clause_idx_list, dataset_cause_emotion_dic, dataset_clauses, dataset_pairs):
        self.dataset_step1 = dataset_step1
        self.dataset_emotion_clause_idx_list = dataset_emotion_clause_idx_list
        self.dataset_emotion_cause_dic = dataset_emotion_cause_dic
        self.dataset_cause_clause_idx_list = dataset_cause_clause_idx_list
        self.dataset_cause_emotion_dic = dataset_cause_emotion_dic
        self.dataset_clauses = dataset_clauses
        self.dataset_pairs = dataset_pairs

    def __getitem__(self, index):
        return self.dataset_step1[index], self.dataset_emotion_clause_idx_list[index], self.dataset_emotion_cause_dic[
            index], self.dataset_cause_clause_idx_list[index], self.dataset_cause_emotion_dic[index], \
            self.dataset_clauses[index], self.dataset_pairs[index]

    def __len__(self):
        return len(self.dataset_step1)


max_input_length = 0
cnt1 = 0
cnt2 = 0
cnt3 = 0


def combine(query_list, clause_list):
    global max_input_length
    global cnt1
    global cnt2
    global cnt3

    input_list = []
    query_span_list = []
    clause_span_list = []
    seg_list = []
    for query in query_list:
        clause_span = []

        query = ' '.join(query).split(' ')  # str -> list
        query_start_idx = 1  # CLS Token
        query_len = len(query)
        query_span_list.append((query_start_idx, query_len))  # [(1,4), (1,8), ...]

        text = ''.join(clause_list)
        text = ' '.join(text).split(' ')  # str -> list
        text_start_idx = query_start_idx + query_len + 1  # SEP Token

        clause_start_idx = text_start_idx
        for clause in clause_list:
            clause_len = len(clause)
            clause_span.append((clause_start_idx, clause_len))
            clause_start_idx += clause_len
        clause_span_list.append(clause_span)  # [[(6,3), (9,5), ...], [(10,4), (14,7), ...], ...]

        input = ['[CLS]'] + query + ['[SEP]'] + text
        input_list.append(input)  # [['[CLS]', '7', '月', '[SEP]', ...], ['[CLS]', '11', '日', '[SEP]', ...], ...]
        seg = [0] * (len(query) + 2) + [1] * len(text)
        seg_list.append(seg)  # [[0, ..., 0, 1, ..., 1], [0, ..., 0, 1, ..., 1], ...]

        max_input_length = len(input) if len(input) > max_input_length else max_input_length
        if len(input) > truncation_length:  # 序列长度大于300的数据样本极少 同时截断处理可能会破坏情感子句或原因子句 故直接舍弃
            cnt1 += 1  # 统计序列长度大于截断阈值的数据样本数量
            input_list.pop()
            seg_list.pop()
            query_span_list.pop()
            clause_span_list.pop()
        cnt2 += 1  # 统计数据样本总数
        if len(input_list) != 0:  # 文本可能只存在一个 <情感子句, 原因子句> 对, pop 后列表为空
            if len(input_list[-1]) > truncation_length:
                cnt3 += 1  # 验证舍弃处理是否有效
    return input_list, seg_list, query_span_list, clause_span_list


def load_train_dataset(dataset_type):
    dataset_name_list = []
    for i in range(1, 2):
        dataset_name_list.append('fold{}'.format(i))

    dataset_total = []
    for dataset_name in dataset_name_list:
        dataset_path = dataset_folder_path + dataset_name + '_' + dataset_type + '.json'
        with open(dataset_path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
            dataset_total += dataset

    dataset_step1 = []
    dataset_step2 = []
    dataset_step3 = []
    dataset_emotion_clause_idx_list = []
    dataset_emotion_cause_dic = []
    dataset_cause_clause_idx_list = []
    dataset_cause_emotion_dic = []
    for doc in dataset_total:
        step1_query_list = []
        step2_query_list = []
        step3_query_list = []

        emotion_clause_idx_list = []  # step1 label
        emotion_cause_dic = defaultdict(list)  # step2 label
        cause_clause_idx_list = []  # step3 label
        cause_emotion_dic = {}

        # static_emotion_query = "Is this an emotional clause?"  # step1
        static_emotion_query = emotion_query_template  # step1
        step1_query_list.append(static_emotion_query)

        doc_len = int(doc['doc_len'])  # 文本子句个数, e.g. 4
        doc_clauses = doc[
            'clauses']  # 文本子句, e.g. [{'clause_id': '1', 'emotion_category': 'null', 'emotion_token': 'null', 'clause': '6月6日'}, {}]
        clause_list = []
        for i in range(doc_len):
            clause_list.append(doc_clauses[i]['clause'])

        doc_pairs = doc['pairs']  # 文本 <情感子句, 原因子句> 对, e.g. [[1,2],[3,4]]
        for pair in doc_pairs:  # 一个情感子句可能对应多个原因子句 一个原因子句对应一个情感子句
            emotion_clause_idx = pair[0]
            cause_clause_idx = pair[1]
            if (emotion_clause_idx - 1) not in emotion_cause_dic.keys():
                emotion_clause_idx_list.append(emotion_clause_idx - 1)
                corresponding_cause_clause_idx_list = []
                for i, j in doc_pairs:
                    if i == emotion_clause_idx:
                        corresponding_cause_clause_idx_list.append(j - 1)
                emotion_cause_dic[emotion_clause_idx - 1] = corresponding_cause_clause_idx_list
                # dynamic_cause_query = clause_list[
                #                           emotion_clause_idx - 1] + 'The corresponding causal clauses for this sentence are?'  # step2
                dynamic_cause_query = clause_list[
                                          emotion_clause_idx - 1] + all_cause_query_template  # step2
                step2_query_list.append(dynamic_cause_query)

            cause_clause_idx_list.append(cause_clause_idx - 1)
            cause_emotion_dic[cause_clause_idx - 1] = emotion_clause_idx - 1
            # dynamic_emotion_query = clause_list[
            #                             cause_clause_idx - 1] + 'Which sentence is the emotional clause corresponding to this sentence?'  # step3
            dynamic_emotion_query = clause_list[
                                        cause_clause_idx - 1] + one_emotion_query_template  # step3

            step3_query_list.append(dynamic_emotion_query)

        # input_list: [['[CLS]', '7', '月', '[SEP]', ...], ['[CLS]', '11', '日', '[SEP]', ...], ...]
        # seg_list: [[0, ..., 0, 1, ..., 1], [0, ..., 0, 1, ..., 1], ...]
        # query_span_list: [(1,4), (1,8), ...]
        # clause_span_list: [[(6,3), (9,5), ...], [(10,4), (14,7), ...], ...]
        step1_input_list, step1_seg_list, step1_query_span_list, step1_clause_span_list = combine(step1_query_list,
                                                                                                  clause_list)
        step2_input_list, step2_seg_list, step2_query_span_list, step2_clause_span_list = combine(step2_query_list,
                                                                                                  clause_list)
        step3_input_list, step3_seg_list, step3_query_span_list, step3_clause_span_list = combine(step3_query_list,
                                                                                                  clause_list)

        if len(step1_input_list) == 0 or len(step2_input_list) == 0 or len(step3_input_list) == 0:  # 保证3step完整训练过程
            continue

        # 数据集中某一文本 3step 的相关信息
        dataset_step1.append((step1_input_list, step1_seg_list, step1_query_span_list, step1_clause_span_list))
        dataset_step2.append((step2_input_list, step2_seg_list, step2_query_span_list, step2_clause_span_list))
        dataset_step3.append((step3_input_list, step3_seg_list, step3_query_span_list, step3_clause_span_list))

        # print(doc["doc_id"])

        # emotion_clause_idx_list: [8, ...]
        # emotion_cause_dic: {8:[9, ...], ...} 一个情感子句可能对应多个原因子句
        dataset_emotion_clause_idx_list.append(emotion_clause_idx_list)
        dataset_emotion_cause_dic.append(emotion_cause_dic)
        assert len(emotion_clause_idx_list) == len(step2_input_list)

        # cause_clause_idx_list: [9, ...]
        # cause_emotion_dic: {9:8} 一个原因子句对应一个情感子句
        dataset_cause_clause_idx_list.append(cause_clause_idx_list)
        dataset_cause_emotion_dic.append(cause_emotion_dic)
        # print("len(cause_clause_idx_list)", len(cause_clause_idx_list))
        # print("len(step3_input_list)", len(step3_input_list))
        # print(doc["doc_id"])
        assert len(cause_clause_idx_list) == len(step3_input_list)
    return Train_Dataset(dataset_step1, dataset_step2, dataset_step3, dataset_emotion_clause_idx_list,
                         dataset_emotion_cause_dic, dataset_cause_clause_idx_list, dataset_cause_emotion_dic)


def load_test_dataset(dataset_type):
    dataset_name_list = []
    for i in range(1, 2):  # 5
        dataset_name_list.append('fold{}'.format(i))

    dataset_total = []
    for dataset_name in dataset_name_list:
        dataset_path = dataset_folder_path + dataset_name + '_' + dataset_type + '.json'
        with open(dataset_path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
            dataset_total += dataset

    dataset_step1 = []
    dataset_emotion_clause_idx_list = []
    dataset_emotion_cause_dic = []
    dataset_cause_clause_idx_list = []
    dataset_cause_emotion_dic = []
    dataset_clauses = []
    dataset_pairs = []
    for doc in dataset_total:
        step1_query_list = []

        emotion_clause_idx_list = []
        emotion_cause_dic = defaultdict(list)
        cause_clause_idx_list = []
        cause_emotion_dic = {}

        # static_emotion_query = "Is this an emotional clause?"  # step1 input
        static_emotion_query = emotion_query_template  # step1 input
        step1_query_list.append(static_emotion_query)

        doc_len = int(doc['doc_len'])
        doc_clauses = doc['clauses']
        clause_list = []
        for i in range(doc_len):
            clause_list.append(doc_clauses[i]['clause'])

        doc_pairs = doc['pairs']  # [[1,2], [3,4]]
        k = []
        for pair in doc_pairs:
            i = pair[0]
            j = pair[1]
            k.append((i - 1, j - 1))
        doc_pairs = k  # [(1,2), (3,4)]
        for pair in doc_pairs:
            emotion_clause_idx = pair[0]
            cause_clause_idx = pair[1]
            if (emotion_clause_idx) not in emotion_cause_dic.keys():
                emotion_clause_idx_list.append(emotion_clause_idx)  # step1 label
                corresponding_cause_clause_idx_list = []
                for i, j in doc_pairs:
                    if i == emotion_clause_idx:
                        corresponding_cause_clause_idx_list.append(j)
                emotion_cause_dic[emotion_clause_idx] = corresponding_cause_clause_idx_list  # step2 label

            cause_clause_idx_list.append(cause_clause_idx)  # step3 label
            cause_emotion_dic[cause_clause_idx] = emotion_clause_idx  # step3 label

        step1_input_list, step1_seg_list, step1_query_span_list, step1_clause_span_list = combine(step1_query_list,
                                                                                                  clause_list)

        if len(step1_input_list) == 0:  # 保证3step完整训练过程
            continue

        dataset_step1.append((step1_input_list, step1_seg_list, step1_query_span_list, step1_clause_span_list))
        dataset_emotion_clause_idx_list.append(emotion_clause_idx_list)
        dataset_emotion_cause_dic.append(emotion_cause_dic)
        dataset_cause_clause_idx_list.append(cause_clause_idx_list)
        dataset_cause_emotion_dic.append(cause_emotion_dic)
        dataset_clauses.append(clause_list)
        dataset_pairs.append(doc_pairs)
    return Test_Dataset(dataset_step1, dataset_emotion_clause_idx_list, dataset_emotion_cause_dic,
                        dataset_cause_clause_idx_list, dataset_cause_emotion_dic, dataset_clauses, dataset_pairs)


train_dataset = load_train_dataset('train')
test_dataset = load_test_dataset('test')


########## 模型构建 ##########
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features  # 768
        self.out_features = out_features  # 192
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)  # [N, in_features] * [in_features, out_features] -> [N, out_features] 特征映射
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1,
                                                                                          2 * self.out_features)  # [N, N, 2 * out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N, N, 1] -> [N, N]

        zero_vec = -9e15 * torch.ones_like(e)  # sigmoid 时 exp(-9e15) 趋近于0
        attention = torch.where(adj > 0, e, zero_vec)  # 原始论文注意力分数由一阶邻居计算
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)  # [N, N] * [N, out_features] -> [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        # nfeat: in_features nhid: out_features
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bert_config = BertConfig.from_pretrained(BERT_PATH)
        self.bert = AutoModel.from_pretrained(BERT_PATH, config=self.bert_config)

        self.attention_linear = nn.Linear(768, 1)

        self.gat = GAT(nfeat=768, nhid=192, dropout=0.1, alpha=0.1, nheads=4)

        self.pre_linear = nn.Linear(768 * 2, 1)

    def attention(self, tensor_group):
        alpha = F.softmax(self.attention_linear(tensor_group), dim=-1)
        attention_tensor = torch.sum(alpha * tensor_group, dim=0)
        return attention_tensor

    def forward(self, step_type, idx, input_ids, attention_mask, token_type_ids, query_span, clause_span,
                emotion_clause_idx_list, emotion_cause_dic, cause_clause_idx_list, cause_emotion_dic):
        # input_ids (1, 300)
        # attention_mask (1, 300)
        # token_type_ids (1, 300)
        # query_span: (1, 8)
        # clause_span: [(10, 4), (14, 11), ...]
        # emotion_clause_idx_list: [9]
        # cause_clause_idx_list: [8]
        # emotion_cause_dic: {9:[8]}
        # cause_emotion_dic: {8:9}
        if step_type == 'step1':
            bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[
                0]
            bert_output = bert_output.squeeze(0)  # 删除 Batch 维度 (300, 768)

            query_span_start, query_span_len = query_span
            query = bert_output[query_span_start:query_span_start + query_span_len]
            query = self.attention(query)  # (768,)

            clause_num = len(clause_span)
            clauses = torch.zeros((clause_num, 768))  # (clause_num, 768)
            for i, (clause_span_start, clause_span_len) in enumerate(clause_span):
                clause = bert_output[clause_span_start:clause_span_start + clause_span_len]
                clause = self.attention(clause)
                clauses[i] = clause

            adj = torch.ones((clause_num, clause_num))  # GAT 原论文中仅计算一阶邻居节点的注意力分数 当前任务中 不确定情感子句和原因子句的相对位置 故计算全局注意力分数

            clauses = self.gat(clauses.to(device), adj.to(device))  # (clause_num, 768)
            queries = query.repeat(clause_num, 1)  # (clause_num, 768)
            combination = torch.cat([queries, clauses], dim=1)  # (clause_num, 768 * 2)

            pre = self.pre_linear(combination).squeeze(-1)
            pre = nn.Sigmoid()(pre)  # 搭配 nn.BCELoss 使用 nn.CrossEntropyLoss 内嵌了 soft-max 函数
            label = torch.zeros(clause_num).to(device)
            for i in emotion_clause_idx_list:
                label[i] = 1

            return pre, label

        if step_type == 'step2':
            bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[
                0]
            bert_output = bert_output.squeeze(0)  # 删除 Batch 维度 (300, 768)

            query_span_start, query_span_len = query_span
            query = bert_output[query_span_start:query_span_start + query_span_len]
            query = self.attention(query)  # (768,)

            corresponding_emotion_clause_idx = emotion_clause_idx_list[idx]
            corresponding_cause_clause_idx_list = emotion_cause_dic[corresponding_emotion_clause_idx]

            clause_num = len(clause_span)
            clauses = torch.zeros((clause_num, 768))  # (clause_num, 768)
            for i, (clause_span_start, clause_span_len) in enumerate(clause_span):
                clause = bert_output[clause_span_start:clause_span_start + clause_span_len]
                clause = self.attention(clause)
                clauses[i] = clause

            adj = torch.ones((clause_num, clause_num))

            clauses = self.gat(clauses.to(device), adj.to(device))  # (clause_num, 768)
            queries = query.repeat(clause_num, 1)  # (clause_num, 768)
            combination = torch.cat([queries, clauses], dim=1)  # (clause_num, 768 * 2)

            pre = torch.sigmoid(self.pre_linear(combination).squeeze(-1))
            label = torch.zeros(clause_num).to(device)
            for i in corresponding_cause_clause_idx_list:
                label[i] = 1

            return pre, label

        if step_type == 'step3':
            bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[
                0]
            bert_output = bert_output.squeeze(0)  # 删除 Batch 维度 (300, 768)

            query_span_start, query_span_len = query_span
            query = bert_output[query_span_start:query_span_start + query_span_len]
            query = self.attention(query)  # (768,)

            try:  # 验证时 step2 预测出的原因子句个数可能大于真的原因子句个数
                corresponding_cause_clause_idx = cause_clause_idx_list[idx]
                corresponding_emotion_clause_idx = cause_emotion_dic[corresponding_cause_clause_idx]
            except:
                corresponding_emotion_clause_idx = 0

            clause_num = len(clause_span)
            clauses = torch.zeros((clause_num, 768))  # (clause_num, 768)
            for i, (clause_span_start, clause_span_len) in enumerate(clause_span):
                clause = bert_output[clause_span_start:clause_span_start + clause_span_len]
                clause = self.attention(clause)
                clauses[i] = clause

            adj = torch.ones((clause_num, clause_num))

            clauses = self.gat(clauses.to(device), adj.to(device))  # (clause_num, 768)
            queries = query.repeat(clause_num, 1)  # (clause_num, 768)
            combination = torch.cat([queries, clauses], dim=1)  # (clause_num, 768 * 2)

            pre = torch.sigmoid(self.pre_linear(combination).squeeze(-1))
            label = torch.zeros(clause_num).to(device)
            label[corresponding_emotion_clause_idx] = 1

            return pre, label


########## 验证过程 ##########
def val(model, test_dataset, tokenizer, epoch):
    total_emotion_precision = 0
    total_emotion_recall = 0
    total_emotion_f1 = 0

    total_cause_precision = 0
    total_cause_recall = 0
    total_cause_f1 = 0

    total_pair_precision = 0
    total_pair_recall = 0
    total_pair_f1 = 0

    model.eval()
    with torch.no_grad():
        test_dataloader = iter(test_dataset)
        test_dataloader_len = len(test_dataset)
        with tqdm(total=test_dataloader_len) as test_bar:
            test_bar.set_description('Epoch %i test' % epoch)

            for step1, emotion_clause_idx_list, emotion_cause_dic, cause_clause_idx_list, cause_emotion_dic, clauses, pairs in test_dataloader:
                # ========================== step1 ==========================
                step_type = 'step1'
                emotion_clause_pre = []
                emotion_precision, emotion_recall, emotion_f1 = 0, 0, 0
                step1_input_list, step1_seg_list, step1_query_span_list, step1_clause_span_list = step1
                for idx, (step1_input, step1_seg, step1_query_span, step1_clause_span) in enumerate(
                        zip(step1_input_list, step1_seg_list, step1_query_span_list, step1_clause_span_list)):
                    # input: ['[CLS]', '7', '月', '[SEP]', ...]
                    # seg: [0, ..., 0, 1, ..., 1]
                    # query_span: (1, 8)
                    # clause_span: [(10, 4), (14, 11), ...]
                    # emotion_clause_idx_list: [9]
                    # cause_clause_idx_list: [8]
                    # emotion_cause_dic: {9:[8]}
                    # cause_emotion_dic: {8:9}
                    ppl_result = tokenizer(step1_input, add_special_tokens=False, padding='max_length', truncation=True,
                                           max_length=truncation_length, is_split_into_words=True)
                    input_ids = torch.tensor(ppl_result['input_ids']).unsqueeze(0)  # 增加 Batch 维度
                    attention_mask = torch.tensor(ppl_result['attention_mask']).unsqueeze(0)
                    step1_seg = step1_seg + [1] * (truncation_length - len(step1_seg))
                    token_type_ids = torch.tensor(step1_seg).unsqueeze(0)

                    pre, label = model(step_type, idx, input_ids.to(device), attention_mask.to(device),
                                       token_type_ids.to(device), step1_query_span, step1_clause_span,
                                       emotion_clause_idx_list, emotion_cause_dic, cause_clause_idx_list,
                                       cause_emotion_dic)
                    label = label.cpu().tolist()
                    pre = pre.cpu().tolist()
                    pre_label = []
                    for i, prob in enumerate(pre):
                        if prob >= step1_threshold:
                            pre_label.append(1)
                            emotion_clause_pre.append(i)
                        else:
                            pre_label.append(0)
                    # emotion_precision, emotion_recall, emotion_f1, _ = precision_recall_fscore_support(label, pre_label)
                    emotion_precision, emotion_recall, emotion_f1, _ = precision_recall_fscore_support(label, pre_label,
                                                                                                       average='binary',
                                                                                                       pos_label=1)

                    # 原代码貌似有误，应该需要缩进
                    # total_emotion_precision += emotion_precision[1]
                    # total_emotion_recall += emotion_recall[1]
                    # total_emotion_f1 += emotion_f1[1]

                    total_emotion_precision += emotion_precision
                    total_emotion_recall += emotion_recall
                    total_emotion_f1 += emotion_f1

                # total_emotion_precision += emotion_precision[0] if len(emotion_precision) == 1 else \
                #     emotion_precision[1]
                # total_emotion_recall += emotion_recall[0] if len(emotion_recall) == 1 else emotion_recall[1]
                # total_emotion_f1 += emotion_f1[0] if len(emotion_f1) == 1 else emotion_f1[1]

                # ========================== step2 ==========================
                step_type = 'step2'
                step2_query_list = []
                emotion_clause_pre_ture = []
                step2_pairs = []
                cause_clause_pre = []
                tmp_cause_precision, tmp_cause_recall, tmp_cause_f1 = 0, 0, 0
                for i in emotion_clause_pre:
                    if i in emotion_clause_idx_list:
                        emotion_clause_pre_ture.append(i)
                diff_num = len(emotion_clause_pre) - len(emotion_clause_pre_ture)

                if len(emotion_clause_pre_ture) == 0:  # 未预测出真的情感子句
                    tmp_cause_precision, tmp_cause_recall, tmp_cause_f1 = 0, 0, 0
                    total_cause_precision += tmp_cause_precision
                    total_cause_recall += tmp_cause_recall
                    total_cause_f1 += tmp_cause_f1
                else:
                    for i in emotion_clause_pre_ture:
                        # dynamic_cause_query = clauses[
                        #                           i] + 'What are the causal clauses corresponding to this sentence?'  # step2
                        dynamic_cause_query = clauses[i] + all_cause_query_template  # step2
                        step2_query_list.append(dynamic_cause_query)

                    step2_input_list, step2_seg_list, step2_query_span_list, step2_clause_span_list = combine(
                        step2_query_list, clauses)

                    for idx, (step2_input, step2_seg, step2_query_span, step2_clause_span) in enumerate(
                            zip(step2_input_list, step2_seg_list, step2_query_span_list, step2_clause_span_list)):
                        ppl_result = tokenizer(step2_input, add_special_tokens=False, padding='max_length',
                                               truncation=True, max_length=truncation_length, is_split_into_words=True)
                        input_ids = torch.tensor(ppl_result['input_ids']).unsqueeze(0)
                        attention_mask = torch.tensor(ppl_result['attention_mask']).unsqueeze(0)
                        step2_seg = step2_seg + [1] * (truncation_length - len(step2_seg))
                        token_type_ids = torch.tensor(step2_seg).unsqueeze(0)

                        pre, label = model(step_type, idx, input_ids.to(device), attention_mask.to(device),
                                           token_type_ids.to(device), step2_query_span, step2_clause_span,
                                           emotion_clause_pre_ture, emotion_cause_dic, cause_clause_idx_list,
                                           cause_emotion_dic)

                        label = label.cpu().tolist()
                        pre = pre.cpu().tolist()
                        pre_label = []

                        for i, prob in enumerate(pre):
                            if prob >= step2_threshold and i not in emotion_clause_pre_ture:
                                pre_label.append(1)
                                step2_pairs.append((emotion_clause_pre_ture[idx], i))
                                if i not in cause_clause_pre:
                                    cause_clause_pre.append(i)
                            else:
                                pre_label.append(0)

                        # cause_precision, cause_recall, cause_f1, _ = precision_recall_fscore_support(label, pre_label)
                        cause_precision, cause_recall, cause_f1, _ = precision_recall_fscore_support(label, pre_label,
                                                                                                     average='binary',
                                                                                                     pos_label=1)

                        # print("cause_precision:", cause_precision)
                        # print("cause_recall:", cause_recall)
                        # print("cause_f1:", cause_f1)

                        # tmp_cause_precision += cause_precision[1]
                        # tmp_cause_recall += cause_recall[1]
                        # tmp_cause_f1 += cause_f1[1]

                        tmp_cause_precision += cause_precision
                        tmp_cause_recall += cause_recall
                        tmp_cause_f1 += cause_f1

                    # step1 可能预测出错误的情感子句 个人认为错误的情感子句预测出的原因子句 哪怕预测正确 也没有意义 故未考虑 导致原因子句的评价指标可能偏低
                    total_cause_precision += tmp_cause_precision // len(emotion_clause_pre_ture)
                    total_cause_recall += tmp_cause_recall // len(emotion_clause_pre_ture)
                    total_cause_f1 += tmp_cause_f1 // len(emotion_clause_pre_ture)

                    # total_cause_precision = tmp_cause_precision
                    # total_cause_recall = tmp_cause_recall
                    # total_cause_f1 = tmp_cause_f1

                # ========================== step3 ==========================
                step_type = 'step3'
                step3_query_list = []
                step3_pairs = []
                pair_precision, pair_recall, pair_f1 = 0, 0, 0
                if len(emotion_clause_pre_ture) == 0:  # 未预测出真的情感子句
                    pair_precision, pair_recall, pair_f1 = 0, 0, 0
                elif len(cause_clause_pre) == 0:  # 未预测出原因子句
                    pair_precision, pair_recall, pair_f1 = 0, 0, 0
                else:
                    for i in cause_clause_pre:
                        # dynamic_emotion_query = clauses[
                        #                             i] + 'Which sentence is the emotional clause corresponding to this sentence?'  # step3
                        dynamic_emotion_query = clauses[i] + one_emotion_query_template  # step3

                        step3_query_list.append(dynamic_emotion_query)

                    step3_input_list, step3_seg_list, step3_query_span_list, step3_clause_span_list = combine(
                        step3_query_list, clauses)

                    for idx, (step3_input, step3_seg, step3_query_span, step3_clause_span) in enumerate(
                            zip(step3_input_list, step3_seg_list, step3_query_span_list, step3_clause_span_list)):
                        ppl_result = tokenizer(step3_input, add_special_tokens=False, padding='max_length',
                                               truncation=True, max_length=truncation_length, is_split_into_words=True)
                        input_ids = torch.tensor(ppl_result['input_ids']).unsqueeze(0)
                        attention_mask = torch.tensor(ppl_result['attention_mask']).unsqueeze(0)
                        step3_seg = step3_seg + [1] * (truncation_length - len(step3_seg))
                        token_type_ids = torch.tensor(step3_seg).unsqueeze(0)

                        pre, _ = model(step_type, idx, input_ids.to(device), attention_mask.to(device),
                                       token_type_ids.to(device), step3_query_span, step3_clause_span,
                                       emotion_clause_idx_list, emotion_cause_dic, cause_clause_idx_list,
                                       cause_emotion_dic)
                        rethink_emotion_clause = torch.argmax(pre).item()
                        step3_pairs.append((rethink_emotion_clause, cause_clause_pre[idx]))

                    # 预测出的 <情感子句, 原因子句> 数量与真的 <情感子句, 原因子句> 数量可能不相符 故无法调sklearn库实现计算
                    final_pairs = []
                    for pair in step2_pairs:
                        if pair in step3_pairs:
                            final_pairs.append(pair)
                        else:
                            if random.random() > step3_threshold:  # rethink 机制为验证成功的 <情感子句, 原因子句> 仍有一定概率保留
                                final_pairs.append(pair)

                    # 手动计算评价指标
                    TP = 0
                    for pair in final_pairs:
                        if pair in pairs:
                            TP += 1
                    pair_precision = TP / (len(final_pairs) + diff_num + 1e-9)  # ×
                    pair_recall = TP / len(pairs)  # √
                    pair_f1 = 2 * pair_precision * pair_recall / (pair_precision + pair_recall + 1e-9)

                total_pair_precision += pair_precision
                total_pair_recall += pair_recall
                total_pair_f1 += pair_f1

                test_bar.update(1)

        # 在此处计算平均值时需要检查分母是否为零
        test_dataloader_len = max(test_dataloader_len, 1)  # 防止除零错误
        average_emotion_precision = total_emotion_precision / test_dataloader_len
        average_emotion_recall = total_emotion_recall / test_dataloader_len
        average_emotion_f1 = total_emotion_f1 / test_dataloader_len

        average_cause_precision = total_cause_precision / test_dataloader_len
        average_cause_recall = total_cause_recall / test_dataloader_len
        average_cause_f1 = total_cause_f1 / test_dataloader_len

        average_pair_precision = total_pair_precision / test_dataloader_len
        average_pair_recall = total_pair_recall / test_dataloader_len
        average_pair_f1 = total_pair_f1 / test_dataloader_len

        print('Test Emotion Precision:', average_emotion_precision, ' Test Emotion Recall:', average_emotion_recall,
              ' Test Emotion F1', average_emotion_f1)
        print('Test Cause Precision:', average_cause_precision, ' Test Cause Recall:', average_cause_recall,
              ' Test Cause F1', average_cause_f1)
        print('Test Pair Precision:', average_pair_precision, ' Test Pair Recall:', average_pair_recall,
              ' Test Pair F1', average_pair_f1)


########## 训练过程 Teacjer Forcing ##########
def train(model, train_dataset, test_dataset):
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)

    train_losses = []

    for epoch in range(1, epochs + 1):
        # start
        total_loss = 0

        model.train()
        # 不同文本 step1 input 序列长度一致 但不同文本的子句个数不同，情感子句的位置不同，因此仍需遍历
        # 不同文本 step2 input 序列长度一致 但不同文本的情感子句个数不同，不同情感子句对应的原因子句个数不同，位置不同，因此仍需遍历
        # 不同文本 step3 input 序列长度一致 但不同文本的原因子句个数不同，位置不同，因此仍需遍历
        # 故为了避免 padding query|子句长度|子句个数 引来冗余信息 降低模型的性能表现 将 batch_size 设置为 1
        # PyTorch 提供的 DataLoader API接口无法处理字符串
        # iter 迭代器没有 shuffle 用法, 考虑到当前不是分类任务 故不存在分类任务没有 shuffle 难以收敛的问题
        # iter 迭代器不支持 len() 方法 故无法自动更新 tqdm
        train_dataloader = iter(train_dataset)
        train_dataloader_len = len(train_dataset)
        with tqdm(total=train_dataloader_len) as train_bar:
            train_bar.set_description('Epoch %i train' % epoch)

            for step1, step2, step3, emotion_clause_idx_list, emotion_cause_dic, cause_clause_idx_list, cause_emotion_dic in train_dataloader:
                model.zero_grad()  # 梯度清零

                # step1
                step_type = 'step1'
                step1_input_list, step1_seg_list, step1_query_span_list, step1_clause_span_list = step1
                for idx, (step1_input, step1_seg, step1_query_span, step1_clause_span) in enumerate(
                        zip(step1_input_list, step1_seg_list, step1_query_span_list, step1_clause_span_list)):
                    # input: ['[CLS]', '7', '月', '[SEP]', ...]
                    # seg: [0, ..., 0, 1, ..., 1]
                    # query_span: (1, 8)
                    # clause_span: [(10, 4), (14, 11), ...]
                    # emotion_clause_idx_list: [9]
                    # cause_clause_idx_list: [8]
                    # emotion_cause_dic: {9:[8]}
                    # cause_emotion_dic: {8:9}
                    ppl_result = tokenizer(step1_input, add_special_tokens=False, padding='max_length', truncation=True,
                                           max_length=truncation_length, is_split_into_words=True)
                    input_ids = torch.tensor(ppl_result['input_ids']).unsqueeze(0)  # 增加 Batch 维度
                    attention_mask = torch.tensor(ppl_result['attention_mask']).unsqueeze(0)
                    step1_seg = step1_seg + [1] * (truncation_length - len(step1_seg))
                    token_type_ids = torch.tensor(step1_seg).unsqueeze(0)

                    pre, label = model(step_type, idx, input_ids.to(device), attention_mask.to(device),
                                       token_type_ids.to(device), step1_query_span, step1_clause_span,
                                       emotion_clause_idx_list, emotion_cause_dic, cause_clause_idx_list,
                                       cause_emotion_dic)
                    step1_loss = criterion(pre, label)

                # step2
                pre_list = []
                label_list = []
                step_type = 'step2'
                step2_input_list, step2_seg_list, step2_query_span_list, step2_clause_span_list = step2
                for idx, (step2_input, step2_seg, step2_query_span, step2_clause_span) in enumerate(
                        zip(step2_input_list, step2_seg_list, step2_query_span_list, step2_clause_span_list)):
                    ppl_result = tokenizer(step2_input, add_special_tokens=False, padding='max_length', truncation=True,
                                           max_length=truncation_length, is_split_into_words=True)
                    input_ids = torch.tensor(ppl_result['input_ids']).unsqueeze(0)
                    attention_mask = torch.tensor(ppl_result['attention_mask']).unsqueeze(0)
                    step2_seg = step2_seg + [1] * (truncation_length - len(step2_seg))
                    token_type_ids = torch.tensor(step2_seg).unsqueeze(0)

                    pre, label = model(step_type, idx, input_ids.to(device), attention_mask.to(device),
                                       token_type_ids.to(device), step2_query_span, step2_clause_span,
                                       emotion_clause_idx_list, emotion_cause_dic, cause_clause_idx_list,
                                       cause_emotion_dic)
                    pre_list.append(pre)
                    label_list.append(label)

                step2_loss = torch.zeros((1)).to(device)
                for pre, label in zip(pre_list, label_list):
                    step2_loss += criterion(pre, label)

                # step3
                pre_list = []
                label_list = []
                step_type = 'step3'
                step3_input_list, step3_seg_list, step3_query_span_list, step3_clause_span_list = step3
                for idx, (step3_input, step3_seg, step3_query_span, step3_clause_span) in enumerate(
                        zip(step3_input_list, step3_seg_list, step3_query_span_list, step3_clause_span_list)):
                    ppl_result = tokenizer(step3_input, add_special_tokens=False, padding='max_length', truncation=True,
                                           max_length=truncation_length, is_split_into_words=True)
                    input_ids = torch.tensor(ppl_result['input_ids']).unsqueeze(0)
                    attention_mask = torch.tensor(ppl_result['attention_mask']).unsqueeze(0)
                    step3_seg = step3_seg + [1] * (truncation_length - len(step3_seg))
                    token_type_ids = torch.tensor(step3_seg).unsqueeze(0)

                    pre, label = model(step_type, idx, input_ids.to(device), attention_mask.to(device),
                                       token_type_ids.to(device), step3_query_span, step3_clause_span,
                                       emotion_clause_idx_list, emotion_cause_dic, cause_clause_idx_list,
                                       cause_emotion_dic)
                    pre_list.append(pre)
                    label_list.append(label)

                step3_loss = torch.zeros((1)).to(device)
                for pre, label in zip(pre_list, label_list):
                    step3_loss += criterion(pre, label)

                loss = (step1_loss + step2_loss + step3_loss) / 2
                # 模型更新
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                train_bar.set_postfix(loss=loss.item())
                train_bar.update(1)

        average_loss = total_loss / train_dataloader_len
        print('Train Loss: ', average_loss)
        train_losses.append(average_loss)

        torch.save(model.state_dict(), "Net_{}.pth".format(epoch + 24))

        val(model, test_dataset, tokenizer, epoch)

    # Plot the training loss curve
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image
    plt.savefig('./Training_loss.png')
    plt.show()


########## 网络模型 ##########
model = Net().to(device)
# model.load_state_dict(torch.load('/root/COLING-ECPE/Net.pth'))

########## 损失函数 + 优化器 ##########
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
training_steps = epochs * len(train_dataset) // batch_size
warmup_steps = int(training_steps * 0.1)
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
                                            num_training_steps=training_steps)

########## 模型训练 + 验证 ##########
train(model, train_dataset, test_dataset)

########## 实验结果分析 ##########
# 1. 现象: 情感子句的评价指标结果与论文接近 原因子句和 <情感子句，原因子句> 对的评价指标结果低于论文
#    原因: 从原始训练集中抽样了十分之一的数据训练模型 测试数据与原始测试集一致
#    分析: 情感子句的判别较为容易
#    Idea: 增加训练数据集 用单任务学习替换多任务学习 聚焦于单个任务
# 2. 现象: 原因子句的预测率低于 <情感子句，原因子句> 对的预测率 二者召回率相近
#    原因: 如 PPT 所示
#    分析: 验证了 rethink 机制的有效性
# 3. 其他Idea: a.用更强大的预训练模型 如 Roberta 等
#              b.双 MRC 框架 可能成本开销与性能提升不匹配 rethink 机制有异曲同工之妙
#              c.token 编码阶段仅编码 query 和唯一子句
#              d.能否同时拥有鸡和蛋 真正实现互相指导
