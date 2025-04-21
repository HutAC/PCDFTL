# -*- codeing = utf-8 -*-
"""
@Time : 2024/10/29
@Author : AC
@File : transformer.py
@Software : PyCharm
"""
import torch
import torch.nn.functional as F

def shift_signal_padding_zero(signal, shift_amount=24):
    shifted_signal = torch.zeros_like(signal)
    if shift_amount > 0:
        shifted_signal[shift_amount:] = signal[:-shift_amount]
    else:
        shifted_signal[:shift_amount] = signal[-shift_amount:]
    return shifted_signal

def time_stretch(signal, factor=1.1):
    if isinstance(signal, torch.Tensor):
        signal = signal.clone().detach().float()
    else:
        signal = torch.tensor(signal, dtype=torch.float32)
    signal = signal.unsqueeze(0).unsqueeze(0)
    length = signal.size(-1)
    new_length = int(length * factor)
    new_signal = F.interpolate(signal, size=new_length, mode='linear', align_corners=False)
    return new_signal.squeeze()

def awgn(data, snr=10.0):
    snr = 10 ** (snr / 10.0)
    xpower = torch.sum(data ** 2) / len(data)
    npower = xpower / snr
    noise = torch.randn(len(data), device=data.device) * torch.sqrt(npower)
    return data + noise
