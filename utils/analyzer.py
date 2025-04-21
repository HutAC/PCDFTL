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
import pandas as pd
import os


# recorder for data
class Recorder:
    def __init__(self, pkl_path, obj):
        self.pkl_path = pkl_path
        self.obj = obj
        save_pkl(obj, pkl_path)
        # obj {"acc": [], "loss":[]}

    def pull(self, key, data):
        datas = read_pkl(self.pkl_path)
        if key not in datas.keys():
            warnings.warn("SYSTEM WARNING : The key not in pkl data keys!")
        datas[key].append(data)
        save_pkl(datas, self.pkl_path)


def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    pass