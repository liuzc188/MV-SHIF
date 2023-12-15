# -*- coding: utf-8 -*-
""" 
    Time    : 2022/4/3 10:57
    Author  : 烟杨绿未成
    FileName: capsule
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MarginLoss(nn.Module):
    """适用于多分类任务"""
    def __init__(self, *, n_labels: int, lambda_: float = 0.5, m_positive: float = 0.9, m_negative: float = 0.1):
        super().__init__()

        self.m_negative = m_negative
        self.m_positive = m_positive
        self.lambda_ = lambda_
        self.n_labels = n_labels

    def forward(self, v: torch.Tensor, labels: torch.Tensor):
        """基于胶囊网络的输出 v"""
        v_norm = torch.sqrt((v ** 2).sum(dim=-1))  # L2 归一化，计算胶囊输出向量的范数

        labels = torch.eye(self.n_labels, device=labels.device)[labels]

        loss = labels * F.relu(self.m_positive - v_norm) + \
               self.lambda_ * (1.0 - labels) * F.relu(v_norm - self.m_negative)
        return loss.sum(dim=-1).mean()


"""
# input
u = torch.randn(8, 4, 200)  # [batch_size, n_capsules, n_features]
print("input is:\n", u)

output = capsule_layer(u)
print("output.shape is:\n", output.shape)
"""
