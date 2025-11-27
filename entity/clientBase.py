# -*- codeing = utf-8 -*-
"""
@Time : 2024/11/11
@Author : AC
@File : clientBase.py
@Software : PyCharm
"""
from model.fed_model import FedModel, FedProModel, FedResModel, FedResModel2
from utils.dataset import HUSTBearing, PUBearing, HUSTGearBox, JNBearing, RealBearing
import torch.nn as nn
import utils.utils as utils
import logging
import copy


class ClientBase(object):
    def __init__(self, args, state, device, return_feature=False, pre=1):
        self.args = args
        self.noise = args.noise
        self.state = state
        self.pre = pre
        self.datasetClass = args.dataset_class
        self.dataset = None
        self.trainLoader, self.valLoader, self.testLoader = None, None, None
        self.target = None
        self.t_loader, self.v_loader, self.te_loader = None, None, None
        self.class_nums = args.class_nums
        self.device = device
        self.fed_dict = None
        self.model = FedResModel(return_feature=return_feature, class_nums=args.class_nums).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.params = [{"params": self.model.parameters(), "lr": self.args.lr}]
        self.lr_scheduler = None
        self.optimizer = None
        self.cd = False
        self.data_dir = self.args.data_dir

    def load_opt(self):
        self.optimizer, self.lr_scheduler = utils.optimizer(self.args, self.params)

    def cross_cd(self):
        self.cd = True

    def cross_target(self):
        self.data_dir = self.args.data_dir2

    def load_data(self):
        if self.datasetClass == 0:
            self.dataset = HUSTBearing(data_dir=self.data_dir,
                                       state=self.state,
                                       noise=self.noise,
                                       pre=self.pre)
        elif self.datasetClass == 1:
            self.dataset = PUBearing(data_dir=self.data_dir,
                                     state=self.state,
                                     pre=self.pre,
                                     noise=self.noise,
                                     cd=self.cd)
        elif self.datasetClass == 2:
            self.dataset = HUSTGearBox(data_dir=self.data_dir,
                                       noise=self.noise,
                                       state=self.state, pre=self.pre)
        elif self.datasetClass == 3:
            self.dataset = JNBearing(data_dir=self.data_dir,
                                     state=self.state, noise=self.noise,
                                     pre=self.pre, cd=self.cd)
        elif self.datasetClass == 4:
            self.dataset = RealBearing(data_dir=self.data_dir,
                                       state=self.state, noise=self.noise,
                                       pre=self.pre)
        self.trainLoader, self.valLoader, self.testLoader = self.dataset.load_data(self.args)

    def download_model(self, serve_model):
        self.model.load_state_dict(copy.deepcopy(serve_model.state_dict()))

    def upload_model(self):
        return self.model

    def info(self, msg):
        logging.info("client {} info : {}".format(self.state, msg))
