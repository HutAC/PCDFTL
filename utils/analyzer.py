# -*- codeing = utf-8 -*-
"""
@Time : 2024/11/6
@Author : AC
@File : analyzer.py
@Software : PyCharm
#  该文件是用于结果分析
"""
import numpy as np
from sklearn.manifold import TSNE
import pickle
import matplotlib.pyplot as plt
import matplotlib
import warnings
import logging
from utils.args import analyze_args, log_args
import torch
import pandas as pd
import os
from utils.dataset import JNBearing, HUSTGearBox, PUBearing
from model.fed_model import FedResModel, LQKF_Model


# 记录者 用于记录训练过程的数据
class Recorder:
    def __init__(self, pkl_path, obj):
        self.pkl_path = pkl_path
        self.obj = obj
        save_pkl(obj, pkl_path)
        # 第一次就要构建好空的obj
        # obj 一般是字典{"acc": [], "loss":[]}

    def pull(self, key, data):
        # 往字典里面放数据
        datas = read_pkl(self.pkl_path)
        if key not in datas.keys():
            warnings.warn("SYSTEM WARNING : The key not in pkl data keys!")
        datas[key].append(data)
        save_pkl(datas, self.pkl_path)

    def val2(self, key, data):
        # 改变某个值 如果它更大的话
        datas = read_pkl(self.pkl_path)
        if key not in datas.keys():
            warnings.warn("SYSTEM WARNING : The key not in pkl data keys!")
        datas[key] = max(data, datas[key])
        save_pkl(datas, self.pkl_path)
# 为什么不直接用字典保存完数据后再用save_pkl?
# 因为我需要每一轮都保存数据 不想重复传参pkl_path
# 而且这么做 一开始就规定好obj的字典格式 让我更清楚


def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_mix(pre, real, class_nums, save_path):
    # 获取混淆矩阵
    # pre: [1, 2, 3, 4] torch.tensor
    # real: [1, 3, 2, 4] torch.tensor
    # label_space: 4
    assert len(pre) == len(real)
    # pre和real的len要一致
    mix = [[0 for t in range(class_nums)] for _ in range(class_nums)]
    for i in range(class_nums):
        mask = real == i
        # 当真的是i的时候
        pre_i = pre[mask]
        # 我们看看pre是什么数字
        for j in pre_i:
            j = j.cpu().item()
            mix[i][j] += 1
        mix[i] = torch.tensor(mix[i])
    mix = torch.stack(mix, dim=0).float()
    sum_mix = torch.sum(mix, dim=1).float()
    for index_m, m in enumerate(mix):
        mix[index_m] = m/sum_mix[index_m]
    txt_mix = mix.numpy().__str__()
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(txt_mix)


def show_fed_cls(pathList):
    color = ["green", "blue", "red", "black"]
    # 创建一个新的图形
    plt.figure()
    for index, path in enumerate(pathList):
        data = read_pkl(path)
        roundData = []
        print(len(data["target_acc"]))
        for i in data["target_acc"]:
            roundData.append(round(i, 3))
        print(data["target_best"])
        plt.plot(roundData, label="line "+str(index), color=color[index])
    # 显示图例
    plt.legend(loc="lower right")
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show()


def ft2(features, labels, path):
    # 使用 t-SNE 将特征降维到 2 维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    # 可视化
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='jet', alpha=0.7)
    plt.colorbar(scatter, label='Class')
    plt.title('Feature Space Visualization (t-SNE)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    # 保存为图片而不是展示
    plt.savefig(path, dpi=300, bbox_inches='tight')  # 保存为高分辨率图片
    plt.close()  # 关闭图形，释放内存


if __name__ == "__main__":
    pass