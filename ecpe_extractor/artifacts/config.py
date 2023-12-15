import os
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

TORCH_SEED = 129


class Config(object):
    def __init__(self):
        # Split
        self.split = 'CV'
        # self.split = 'TVT'

        # PLM
        # self.bert_cache_path = 'cyclone/simcse-chinese-roberta-wwm-ext'  # chinese
        # self.bert_cache_path = 'bert-base-chinese'
        # self.bert_cache_path = 'princeton-nlp/sup-simcse-roberta-large'# english
        # self.bert_cache_path = '/home/ubuntu/workspace1/rmj/ECPE/ecpe_extractor/artifacts/simcse'# english
        # self.bert_cache_path = '/home/ubuntu/workspace1/rmj/ECPE/ecpe_extractor/artifacts/bert'# english
        # self.bert_cache_path = '/home/ubuntu/workspace1/rmj/ECPE/ecpe_extractor/artifacts/longformer'  # english
        self.bert_cache_path = "/home/ubuntu/workspace1/rmj/LJP/PLM/nezha"

        # Training
        self.epochs = 30
        self.batch_size = 3  # 1约占14G显存
        self.lr = 1e-5
        self.tuning_bert_rate = 1e-5
        self.gradient_accumulation_steps = 1
        self.dp = 0.1
        self.warmup_proportion = 0.1
        self.sentiment_category = {'null': 0, 'happiness': 1, 'sadness': 2, 'disgust': 3, 'surprise': 4, 'fear': 5,
                                   'anger': 6}

        # GNN
        self.feat_dim = 768
        self.gnn_dims = '192'
        self.att_heads = '4'

        # Capsule
        self.in_caps = 21
        self.out_caps = 21
        self.in_d = 1536
        self.out_d = 1536
        self.iterations = 5
