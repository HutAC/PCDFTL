# -*- codeing = utf-8 -*-
"""
@Time : 2024/11/30
@Author : AC
@File : serve.py
@Software : PyCharm
"""
from .serveBase import ServeBase
import torch
import utils.utils as utils
from loss.mkmmd import MKMMD
from loss.lmmd import LMMD_loss
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import copy

# There's no server in this framework,
# it's just what the code calls it
class Serve(ServeBase):
    def __init__(self, args, clients, source_nums, device, model_dir, index, recorder=None):
        super(Serve, self).__init__(args, clients, source_nums, device)
        self.recorder = recorder
        self.index = index
        self.source_client = self.source_clients[0]
        self.target_client = self.target_clients[0]
        self.fed = None
        self.source_best = 0
        self.model_dir = os.path.join(model_dir, args.log_file.replace(".log", ""))
        self.now_epoch = None

    def train(self, now_epoch, freeze=True):
        self.now_epoch = now_epoch
        s_train = self.source_client.trainLoader
        t_train = self.target_client.trainLoader
        self.source_client.train(now_epoch, self.fed, freeze=freeze)
        loss_fn = nn.CrossEntropyLoss()
        device = self.device
        # Simulating upload features
        # I didn't use the dataset directly after uploading it
        # ==> Equivalent to uploading features
        epoch = self.args.serve_epoch
        model = self.source_client.model
        opt = self.source_client.opt2
        lr_s = self.source_client.lr_s2
        opt_h = None
        lr_s_h = None
        model_h = None
        if not freeze:
            opt_h = self.source_client.opt
            lr_s_h = self.source_client.lr_s
            model_h = copy.deepcopy(model.feature)
            utils.lock_grad(model_h)
        for e in range(epoch):
            cls_loss = []
            mmd_loss = []
            accList = []
            for i, ((x1, y1), (tx, ty)) in enumerate(zip(*[s_train, t_train])):
                if freeze:
                    model.eval()
                    with torch.no_grad():
                        x1, y1 = x1.to(device), y1.to(device)
                        x1 = x1.permute(0, 2, 1)
                        x1 = x1.unsqueeze(-1)

                        tx, ty = tx.to(device), ty.to(device)
                        tx = tx.permute(0, 2, 1)
                        tx = tx.unsqueeze(-1)

                        clss, fs, _ = model(x1)
                        clst, ft, _ = model(tx)
                        acc = utils.accuracy(clst, ty)
                        acc = round(acc, 2)
                        accList.append(acc)
                else:
                    model_h.eval()
                    with torch.no_grad():
                        tx, ty = tx.to(device), ty.to(device)
                        tx = tx.permute(0, 2, 1)
                        tx = tx.unsqueeze(-1)
                        clst, _, _ = model(tx)
                        ft = model_h(tx)
                        acc = utils.accuracy(clst, ty)
                        acc = round(acc, 2)
                        accList.append(acc)
                    x1, y1 = x1.to(device), y1.to(device)
                    x1 = x1.permute(0, 2, 1)
                    x1 = x1.unsqueeze(-1)
                    clss, fs, _ = model(x1)
                opt.zero_grad()
                if not freeze:
                    opt_h.zero_grad()
                    model.feature.train()
                model.pro.train()
                model.cls.train()
                if freeze:
                    fs = fs.detach()
                    ft = ft.detach()
                f_s = model.pro(fs)
                f_t = model.pro(ft)

                c_s = model.cls(f_s)
                c_t = model.cls(f_t)

                mmd = MKMMD(f_s, f_t)
                # mmd = self.lmmder.get_loss(f_s, f_t, y1.long(), F.softmax(c_t.detach(), dim=1), self.class_nums)
                cls = loss_fn(c_s, y1.long())
                loss = cls + 0.5 * mmd
                loss.backward()
                opt.step()
                if not freeze:
                    opt_h.step()
                cls_loss.append(cls.detach().cpu())
                mmd_loss.append(mmd.detach().cpu())
            avg_cls = np.mean(cls_loss)
            avg_mmd = np.mean(mmd_loss)
            avg_acc = np.mean(accList)
            lr_s.step()
            if not freeze:
                lr_s_h.step()
            Serve.info("(client {}) train {}/{} : acc : {}  cls: {}  mmd : {}".format(self.source_client.state, e, epoch, avg_acc, avg_cls, avg_mmd))
            tacc_key = "client{}_tacc".format(self.index)
            self.recorder.pull(tacc_key, avg_acc)

    def get_fed(self):
        device = self.device
        model = self.source_client.model
        t_train = self.target_client.trainLoader
        model.eval()
        with torch.no_grad():
            t_f = []
            t_l = []
            accList = []
            for tx, ty in t_train:
                tx, ty = tx.to(device), ty.to(device)
                tx = tx.permute(0, 2, 1)
                tx = tx.unsqueeze(-1)
                clxt, ft, ht = model(tx)
                t_f.extend(ht.detach())
                pre = torch.max(clxt.cpu(), 1)[1].numpy()
                pre = torch.tensor(pre)
                t_l.extend(pre)
                acc = utils.accuracy(clxt, ty)
                accList.append(acc)
            avg_acc = np.mean(accList)
            if avg_acc > self.source_best:
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                self.source_best = avg_acc
                state = self.source_client.state
                # state-best
                best_model_file = state+"-best"+".tar"
                model_path = os.path.join(self.model_dir, best_model_file)
                torch.save({
                    "model": model.state_dict(),
                    "state": state,
                    "acc": self.source_best,
                    "epoch": self.now_epoch,
                    "total_epoch": self.args.epoch
                }, model_path)
            Serve.info("client {} Fed Infer : acc : {}".format(self.source_client.state, avg_acc))
            t_f = torch.stack(t_f, dim=0)
            t_l = torch.stack(t_l, dim=0)
            self.fed = utils.getFedByLabel(t_f, t_l.long())