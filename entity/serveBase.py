# -*- codeing = utf-8 -*-
"""
@Time : 2024/11/11
@Author : AC
@File : serveBase.py
@Software : PyCharm
"""
import torch
import logging
import copy


class ServeBase(object):
    def __init__(self, args, clients, source_nums, device):
        self.args = args
        self.clients = clients
        self.class_nums = args.class_nums
        self.device = device
        self.g_model = None
        self.fed_dict = None
        self.source_nums = source_nums
        self.client_models = []
        self.source_clients = self.clients[:self.source_nums]
        self.target_clients = self.clients[self.source_nums:]
        self.source_weights = torch.full((self.source_nums, 1), 1 / self.source_nums).view(-1).float()
        self.source_mmd_weights = None
        self.source_ctr_weights = None
        self.upload_features = [[] for _ in range(len(self.clients))]
        self.upload_labels = [[] for _ in range(len(self.clients))]

    def send_model(self):
        for client in self.clients:
            client.download_model(self.g_model)

    def receive_model(self):
        self.client_models = []
        for source_client in self.source_clients:
            self.client_models.append(source_client.upload_model())

    def aggregate_parameters(self):
        assert len(self.client_models) > 0
        self.g_model = copy.deepcopy(self.client_models[0]).to(self.device)
        g_model_state_dict = self.g_model.state_dict()
        for key in g_model_state_dict:
            g_model_state_dict[key].zero_().float()
        for w, client_model in zip(self.source_weights, self.client_models):
            client_model_state_dict = client_model.state_dict()
            for key in g_model_state_dict:
                g_model_state_dict[key] = g_model_state_dict[key].float() + client_model_state_dict[key].clone().float() * w
        self.g_model.load_state_dict(g_model_state_dict)

    @staticmethod
    def info(msg):
        logging.info("serve info : {}".format(msg))
