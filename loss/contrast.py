# -*- codeing = utf-8 -*-
"""
@Time : 2024/10/29
@Author : AC
@File : 1.py
@Software : PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# SimCLR官方的
# 用第二个对比损失后面需要用交叉熵损失
class ContrastiveLoss(nn.Module):
    def __init__(self, args, device, n_views=2, tp=0.07):
        super().__init__()
        self.args = args
        self.device = device
        self.n_views = n_views
        self.temperature = tp

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

