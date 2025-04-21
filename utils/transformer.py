# -*- codeing = utf-8 -*-
"""
@Time : 2024/10/29
@Author : AC
@File : transformer.py
@Software : PyCharm
"""
import torch
import torch.nn.functional as F


# 信号平移
def shift_signal_padding_zero(signal, shift_amount=24):
    """
    将信号在时间轴上平移，并用零填充空缺位置（使用 PyTorch）。

    参数:
    signal (torch.Tensor): 原始振动信号
    shift_amount (int): 平移的样本数（正值向右平移，负值向左平移）

    返回:
    torch.Tensor: 平移后的信号
    """
    shifted_signal = torch.zeros_like(signal)  # 创建与原信号同样大小的零张量
    if shift_amount > 0:
        shifted_signal[shift_amount:] = signal[:-shift_amount]  # 向右平移
    else:
        shifted_signal[:shift_amount] = signal[-shift_amount:]  # 向左平移
    return shifted_signal


# 随机时间伸缩
def time_stretch(signal, factor=1.1):
    """时间伸缩变换"""
    if isinstance(signal, torch.Tensor):
        signal = signal.clone().detach().float()  # 如果是张量，使用 clone().detach()
    else:
        signal = torch.tensor(signal, dtype=torch.float32)  # 如果是 NumPy 数组或其他类型，转为张量

    signal = signal.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
    length = signal.size(-1)
    new_length = int(length * factor)
    new_signal = F.interpolate(signal, size=new_length, mode='linear', align_corners=False)
    return new_signal.squeeze()


# 添加高斯白噪声
def awgn(data, snr=10.0):
    """
    添加高斯白噪声
    """
    snr = 10 ** (snr / 10.0)
    xpower = torch.sum(data ** 2) / len(data)
    npower = xpower / snr
    noise = torch.randn(len(data), device=data.device) * torch.sqrt(npower)
    return data + noise
