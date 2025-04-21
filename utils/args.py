# -*- codeing = utf-8 -*-
"""
@Time : 2024/10/26
@Author : AC
@File : args.py
@Software : PyCharm
"""
import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    # 基础训练参数
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_scheduler", type=str, default="cos")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--local_epoch", type=int, default=1)
    parser.add_argument("--serve_epoch", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    # 数据集参数
    parser.add_argument("--data_dir", type=str, default=r"E:\LZK\codeWork\python\items\AcademicResearch\my\dataset\HUST-bearing-dataset\Rawdata2")
    parser.add_argument("--data_dir2", type=str, default=None)
    parser.add_argument("--dataset_class", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--class_nums", type=int, default=10)
    # 其他参数
    parser.add_argument("--log_file", type=str, default="temp.log")
    parser.add_argument("--save_epoch", type=int, default=10)
    parser.add_argument("--lamda", type=float, default=1.0)
    parser.add_argument("--mmd_w", type=float, default=0.5, help="MK-MMD weights")
    parser.add_argument("--bias", type=float, default=2.0, help="计算权重时的平滑因子，增大该因子可以增大稳定性")
    parser.add_argument("--weight_w", type=float, default=0.9, help="计算权重时的差异化权重的比例")
    parser.add_argument("--p_way", type=str, default="mean", choices=["mean", "cos"])
    parser.add_argument("--task", type=str, default="T1")
    # 策略启用参数
    parser.add_argument("--calc_w", type=bool, default=True)
    parser.add_argument("--freeze", type=bool, default=True)
    return parser.parse_args()


def analyze_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--class_nums", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dataset_class", type=int, default=1)
    parser.add_argument("--result_name", type=str, default="RDFTL")
    parser.add_argument("--data_dir", type=str, default="E:\LZK\codeWork\python\items\AcademicResearch\my\dataset\PU-bearing-dataset-min")
    parser.add_argument("--model_path", type=str, default="E:\LZK\codeWork\python\items\AcademicResearch\my\logs_real\RDFTL\T4\RDFTL-best.tar")
    parser.add_argument("--model_type", type=int, default=1)
    parser.add_argument("--state", type=str, default="3")
    parser.add_argument("--way", type=str, default="tsne")
    return parser.parse_args()


def log_args(args):
    # save the args
    for k, v in args.__dict__.items():
        if len(str(v)) > 0:
            logging.info("{}: {}".format(k, v))
