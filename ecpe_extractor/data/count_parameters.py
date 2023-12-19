# @Time : 2023/12/8 16:43
# @Author : Cheng Yang
# @File ：count_parameters.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Squash(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, s: torch.Tensor):  # s: [batch_size, n_capsules, n_features]
        s2 = (s ** 2).sum(dim=-1, keepdims=True)
        return (s2 / (1 + s2)) * (s / torch.sqrt(s2 + self.epsilon))


class Router(nn.Module):
    def __init__(self, in_caps: int, out_caps: int, in_d: int, out_d: int, iterations: int):  # int_d: 前一层胶囊的特征数目
        super().__init__()
        self.in_caps = in_caps  # 胶囊数目
        self.out_caps = out_caps
        self.iterations = iterations
        self.softmax = nn.Softmax(dim=1)
        self.squash = Squash()

        # maps each capsule in the lower layer to each capsule in this layer
        self.weight = nn.Parameter(torch.randn(in_caps, out_caps, in_d, out_d), requires_grad=True)

    def forward(self, u: torch.Tensor):  # 低层胶囊的输入
        """
        input(s) shape: [batch_size, n_capsules, n_features]
        output shape: [batch_size, n_capsules, n_features]
        """

        u_hat = torch.einsum('ijnm,bin->bijm', self.weight, u)
        b = u.new_zeros(u.shape[0], self.in_caps, self.out_caps)
        v = None
        for i in range(self.iterations):
            c = self.softmax(b)
            s = torch.einsum('bij,bijm->bjm', c, u_hat)
            v = self.squash(s)
            a = torch.einsum('bjm,bijm->bij', v, u_hat)
            b = b + a
        return v


model = Router(4, 2, 200, 100, 2)


class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, hidden_size):
        super(TransformerEncoderModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        encoder_layers = TransformerEncoderLayer(embed_size, heads, hidden_size)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.transformer_encoder(embedded)
        return output


# 模型参数
vocab_size = 1000  # 假设词汇量大小为1000
embed_size = 512
num_layers = 6
heads = 8
hidden_size = 768

# 创建模型实例
transformer_model = TransformerEncoderModel(vocab_size, embed_size, num_layers, heads, hidden_size)


# 计算模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"模型参数量：{count_parameters(transformer_model)}")
# capsule_model: 160000
#  transformer_model: 11554304


"""
# input
u = torch.randn(8, 4, 200)  # [batch_size, n_capsules, n_features]
print("input is:\n", u)

output = capsule_layer(u)
print("output.shape is:\n", output.shape)
"""
