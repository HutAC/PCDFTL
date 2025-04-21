# -*- codeing = utf-8 -*-
"""
@Time : 2024/11/30
@Author : AC
@File : PCDFTL.py
@Software : PyCharm
"""
from entity.client import Client
from entity.serve import Serve
import os
from utils.logger import setlogger
from utils.args import parse_args, log_args
from utils.utils import check_dir
import warnings
import torch
import logging
import copy
from utils.analyzer import Recorder
from task import get_state

# 获取文件名（不包含路径）
file_name = os.path.basename(__file__).replace(".py", "")

modelSaveDir = os.path.join("./checkPoint", file_name)
logSaveDir = os.path.join("./logs", file_name)


def main(args):
    # Consider the gpu or cpu condition
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
        logging.info('using {} gpus'.format(device_count))
        assert args.batch_size % device_count == 0, "batch size should be divided by device count"
    else:
        warnings.warn("gpu is not available")
        device = torch.device("cpu")
        device_count = 1
        logging.info('using {} cpu'.format(device_count))
    state = get_state(args.task)
    # if args.dataset_class == 0:
    #     state = ["20", "25", "35", "60"]
    # elif args.dataset_class == 1:
    #     state = ["0", "1", "2", "3"]
    # elif args.dataset_class == 2:
    #     state = ["20_1", "20_2", "25_1", "25_2"]
    # elif args.dataset_class == 3:
    #     state = ["600", "800", "None", "1000"]
    # elif args.dataset_class == 4:
    #     state = ["600", "800", "1000", "20"]
    # elif args.dataset_class == 5:
    #     state = ["20", "25", "30", "600"]
    state_file_name = state[0] + "_" + state[1] + "_" + state[2] + "-" + state[3]
    pkl_save = {
        "client1_ctr": [],  # 客户端1的对比损失的变化
        "client2_ctr": [],
        "client3_ctr": [],
        "client1_mmd": [],  # 客户端1的MK-MMD的变化
        "client2_mmd": [],
        "client3_mmd": [],
        "client1_tacc": [],  # 客户端1在目标域上的准确率变化
        "client2_tacc": [],
        "client3_tacc": [],
        "target_acc": [],  # 目标域的准确率列表
        "target_best": 0   # 目标域最好的准确率
    }
    pkl_file_name = state_file_name + args.log_file.replace(".log", "") + ".pkl"
    pkl_path = os.path.join(logSaveDir, pkl_file_name)
    recorder = Recorder(pkl_path, pkl_save)

    logging.info("SYSTEM INFO: Fed Task : {}".format(state_file_name))
    if args.dataset_class != 3:
        args1 = copy.deepcopy(args)
        args2 = copy.deepcopy(args)
        args3 = copy.deepcopy(args)
        args4 = copy.deepcopy(args)
        cd_flag = 0
        if args.dataset_class == 4:
            args1.dataset_class = 3
            args2.dataset_class = 3
            args3.dataset_class = 3
            args4.dataset_class = 1
            cd_flag = 1
        if args.dataset_class == 5:
            args1.dataset_class = 1
            args2.dataset_class = 1
            args3.dataset_class = 1
            args4.dataset_class = 3
            cd_flag = 1
        client1 = Client(args=args1, device=device, state=state[0], model_dir=modelSaveDir, recorder=recorder,
                         index=1)
        client2 = Client(args=args2, device=device, state=state[1], model_dir=modelSaveDir, recorder=recorder,
                         index=2)
        client3 = Client(args=args3, device=device, state=state[2], model_dir=modelSaveDir, recorder=recorder,
                         index=3)
        client4 = Client(args=args4, device=device, state=state[3], model_dir=modelSaveDir, recorder=recorder,
                         index=4)
        if cd_flag == 1:
            client1.cross_cd()
            client2.cross_cd()
            client3.cross_cd()
            client4.cross_cd()
            client4.cross_target()
        client1.load_data()
        client2.load_data()
        client3.load_data()
        client4.load_data()
        clients = [client1, client2, client3]
        s1 = [client1, client4]
        s2 = [client2, client4]
        s3 = [client3, client4]

        serve1 = Serve(args=args, device=device, clients=s1, source_nums=1, model_dir=modelSaveDir, recorder=recorder, index=1)
        serve2 = Serve(args=args, device=device, clients=s2, source_nums=1, model_dir=modelSaveDir, recorder=recorder, index=2)
        serve3 = Serve(args=args, device=device, clients=s3, source_nums=1, model_dir=modelSaveDir, recorder=recorder, index=3)

        serves = [serve1, serve2, serve3]
    else:
        client1 = Client(args=args, device=device, state=state[0], model_dir=modelSaveDir, recorder=recorder, index=1)
        client2 = Client(args=args, device=device, state=state[1], model_dir=modelSaveDir, recorder=recorder, index=2)
        client4 = Client(args=args, device=device, state=state[3], model_dir=modelSaveDir, recorder=recorder, index=3)
        client1.load_data()
        client2.load_data()
        client4.load_data()
        clients = [client1, client2]
        s1 = [client1, client4]
        s2 = [client2, client4]
        serve1 = Serve(args=args, device=device, clients=s1, source_nums=1, model_dir=modelSaveDir, recorder=recorder,
                       index=1)
        serve2 = Serve(args=args, device=device, clients=s2, source_nums=1, model_dir=modelSaveDir, recorder=recorder,
                       index=2)
        serves = [serve1, serve2]
    epoch = args.epoch
    for e in range(epoch):
        logging.info("SYSTEM INFO : {} / {}".format(e, epoch))
        for serve in serves:
            serve.train(e, freeze=args.freeze)
            serve.get_fed()
        if e > 0:
            client4.target_val(clients=clients, now_epoch=e, calc_w=args.calc_w)


if __name__ == "__main__":
    check_dir()
    if not os.path.exists(modelSaveDir):
        os.makedirs(modelSaveDir)
    if not os.path.exists(logSaveDir):
        os.makedirs(logSaveDir)
    args = parse_args()
    log_path = os.path.join(logSaveDir, args.log_file)
    setlogger(log_path)
    log_args(args)
    main(args)
