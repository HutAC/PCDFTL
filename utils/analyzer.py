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


if __name__ == "__main__":
    pass