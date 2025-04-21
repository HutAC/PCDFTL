# -*- codeing = utf-8 -*-
"""
@Time : 2024/11/30
@Author : AC
@File : client.py
@Software : PyCharm
"""
from .clientBase import ClientBase
from loss.mkmmd import MKMMD
from loss.lmmd import LMMD_loss
import torch
import utils.utils as utils
import torch.nn.functional as F
import numpy as np
import logging
import os


class Client(ClientBase):
    def __init__(self, args, device, state, model_dir, index, return_feature=True, pre=1, recorder=None):
        super(Client, self).__init__(args=args,
                                     device=device,
                                     state=state,
                                     return_feature=return_feature,
                                     pre=pre
                                     )
        self.index = index
        self.now_epoch = None
        self.lmmder = LMMD_loss()
        self.recorder = recorder  # 记录者
        self.model_dir = os.path.join(model_dir, args.log_file.replace(".log", ""))
        self.target_best = 0  # 用来记录目标域客户端推理的最好准确率
        self.ctr_loss = 0  # 用来记录这个客户端每次本地训练的对比损失
        self.mmd_loss = 0  # 用来记录这个客户端的差异化损失
        self.ctr_var = 0
        self.mmd_var = 0
        self.ctr_loss_list = []
        self.mmd_loss_list = []
        self.mmd_w = 0.5
        self.ctr_w = 1 - self.mmd_w
        self.optimizer, self.lr_scheduler = None, None  # 清空原来的优化器
        self.opt, self.lr_s = utils.optimizer(args, [{"params": self.model.feature.parameters(), "lr": self.args.lr}])
        self.opt2, self.lr_s2 = utils.optimizer(
            args,
            [
                {"params": self.model.pro.parameters(), "lr": args.lr},
                {"params": self.model.cls.parameters(), "lr": args.lr}
            ]
        )

    def train(self, now_epoch, fed=None, freeze=True):
        self.now_epoch = now_epoch
        self.model.feature.train()
        epoch = self.args.local_epoch
        for e in range(epoch):
            saccList = []
            self.ctr_loss_list = []
            self.mmd_loss_list = []
            for x1, y1 in self.trainLoader:
                self.opt.zero_grad()
                if not freeze and fed is not None:
                    # 如果不采用我的策略 就要优化后面的
                    # 同时要在有fed的时候
                    self.opt2.zero_grad()
                x1, y1 = x1.to(self.device), y1.to(self.device)
                x1 = x1.permute(0, 2, 1)
                x1 = x1.unsqueeze(-1)

                clx1, f1, h1 = self.model(x1)
                cls = self.loss_fn(clx1, y1.long())
                loss = cls
                if fed is not None:
                    # ---- MK-MMD ----
                    choose_f = []
                    t_new = []
                    choose_label = []
                    for index, f in enumerate(h1):
                        key = int(y1[index].item())
                        if key in fed.keys():
                            choose_f.append(f)
                            t_new.append(fed[key])
                            choose_label.append(y1[index])
                    choose_f = torch.stack(choose_f, dim=0).to(self.device)
                    t_new = torch.stack(t_new, dim=0).to(self.device)
                    choose_label = torch.stack(choose_label, dim=0).view(-1).to(self.device)
                    mmd = MKMMD(choose_f, t_new)
                    # mmd = self.lmmder.get_loss(choose_f, t_new, choose_label.long(), choose_label.long(), self.class_nums)
                    loss += self.mmd_w * mmd
                    self.mmd_loss_list.append(mmd.detach().cpu())
                    # ---- END ----

                    # ---- 对比损失 ----
                    global_protos_emb = []
                    # 首先要拿到fed的key
                    # 然后判断一下有没有class_nums那么多个
                    # 要先排序一下
                    target_fed_keys = list(fed.keys())
                    # 就不要用源域的原型了，直接用0值填充 因为0*任何数都为0 所以相似度也是0
                    for k in range(self.args.class_nums):
                        if k in target_fed_keys:
                            global_protos_emb.append(fed[k])
                        else:
                            global_protos_emb.append(torch.zeros((fed[target_fed_keys[0]].shape[0],)).to(self.device))
                    global_protos_emb = torch.stack(global_protos_emb).to(self.device)
                    h1 = F.normalize(h1, dim=1)
                    global_protos_emb = F.normalize(global_protos_emb, dim=1)
                    similarity = torch.matmul(h1, global_protos_emb.T)/0.1
                    cn = self.loss_fn(similarity, y1.long())
                    loss += self.ctr_w * cn
                    self.ctr_loss_list.append(cn.detach().cpu())
                    # ---- END ----
                loss.backward()
                self.opt.step()
                if not freeze and fed is not None:
                    # 如果不采用我的策略 就要优化后面的
                    self.opt2.step()
                sacc = utils.accuracy(clx1, y1)
                saccList.append(sacc)
            self.lr_s.step()
            avg_sacc = np.mean(saccList)
            self.ctr_loss = np.mean(self.ctr_loss_list) if len(self.ctr_loss_list) > 0 else 0
            self.ctr_var = np.var(self.ctr_loss_list, axis=0) if len(self.ctr_loss_list) > 0 else 0
            self.mmd_loss = np.mean(self.mmd_loss_list) if len(self.mmd_loss_list) > 0 else 0
            self.mmd_var = np.var(self.mmd_loss_list, axis=0) if len(self.mmd_loss_list) > 0 else 0
            self.info("train {}/{} sacc : {}  ctr : {}  mkmmd : {}".format(e, epoch, avg_sacc, self.ctr_loss, self.mmd_loss))

            mmd_key = "client{}_mmd".format(self.index)
            ctr_key = "client{}_ctr".format(self.index)
            self.recorder.pull(mmd_key, self.mmd_loss)
            self.recorder.pull(ctr_key, self.ctr_loss)

    # 目标域客户端测试用的函数
    def target_val(self, clients, now_epoch, calc_w=True):
        # calc_w : 是否计算权重 还是说直接均值
        # weight_w = 0.9
        weight_w = self.args.weight_w
        bias = self.args.bias  # 平滑因子
        eps = 1e-5  # 平滑值
        self.now_epoch = now_epoch
        # 根据源域客户端的mmd的均值与方差作为不同客户端模型的话语权
        # ---- 收集源域客户端数据 -----
        models = []
        mmds = []
        mmd_vars = []
        for client in clients:
            models.append(client.model)
            mmds.append(client.mmd_loss)
            mmd_vars.append(client.mmd_var)
        # ---- END -----

        # ---- 计算源域客户端话语权 ----
        if calc_w:
            mmds = torch.tensor(mmds)
            mmd_vars = torch.tensor(mmd_vars)
            # mmds = mmds/torch.sum(mmds, dim=0)
            mmd_vars = mmd_vars/torch.sum(mmd_vars, dim=0)
            scores = 1/(mmds*mmd_vars*bias)
            weights = F.softmax(scores, dim=0).to(self.device)
        else:
            weights = torch.full((len(clients), 1), 1 / len(clients)).view(-1).float().to(self.device)
        logging.info("infer mix weights : {}".format(weights))
        # 广播 weights 成为 (n, 1, 1) 的形状以便与 outputs 相乘
        weights = weights.view(len(models), 1, 1)  # 转换为 (n, 1, 1)
        if calc_w:
            weights = weights.repeat(1, self.args.batch_size, 1)
        # ---- END ----
        # ---- 不同模型推理并聚合推理结果 ----
        correct = 0
        all_s = 0
        for x, y in self.trainLoader:
            x, y = x.to(self.device), y.to(self.device)
            x = x.permute(0, 2, 1)
            x = x.unsqueeze(-1)

            infers = []
            var_weights = []
            for model in models:
                cls, _, _ = model(x)
                cls = F.softmax(cls, dim=1)
                # cls [B, C]
                if calc_w:
                    # 换成熵
                    var = 1/(-torch.sum(cls.detach() * torch.log(cls.detach() + eps), dim=1)+eps)
                    var_weights.append(var)
                infers.append(cls)
            infers = torch.stack(infers, dim=0)
            if calc_w:
                var_weights = torch.stack(var_weights, dim=0)
                # [3, B]
                var_weights = var_weights/torch.sum(var_weights, dim=0)
                # print(var_weights.shape)  # [3, 128]
                var_weights = var_weights.view(len(models), var_weights.size(1), 1)
                # print(weights.shape) # [3, 128, 1]
                # print(var_weights.shape) # [3, 128, 1]
                weights = weights*weight_w + var_weights*(1-weight_w)

            # 将每个模型的推理结果乘以对应的权重
            weighted_outputs = infers * weights

            # 对加权后的结果进行求和，得到最终的加权推理结果，形状为 (b, c)
            final_output = weighted_outputs.sum(dim=0)
            predicted_classes = final_output.argmax(dim=1)
            correct += (predicted_classes == y.long()).sum().item()  # 计算预测正确的样本数量
            all_s += y.size(0)
        acc = correct/all_s
        acc = round(acc*100, 2)
        if acc > self.target_best:
            self.target_best = acc
            model_file = "best-mix.tar"
            model_path = os.path.join(self.model_dir, model_file)
            if len(models) == 3:
                torch.save({
                    "model1": models[0].state_dict(),
                    "model2": models[1].state_dict(),
                    "model3": models[2].state_dict(),
                    "weights": weights.cpu(),
                    "acc": acc,
                    "epoch": self.now_epoch,
                    "total_epoch": self.args.epoch
                }, model_path)
            else:
                torch.save({
                    "model1": models[0].state_dict(),
                    "model2": models[1].state_dict(),
                    "model3": None,
                    "weights": weights.cpu(),
                    "acc": acc,
                    "epoch": self.now_epoch,
                    "total_epoch": self.args.epoch
                }, model_path)
        if self.now_epoch % self.args.save_epoch == 0:
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            model_file = str(self.now_epoch)+"-mix-" + str(self.now_epoch) + ".tar"
            model_path = os.path.join(self.model_dir, model_file)
            if len(models) == 3:
                torch.save({
                    "model1": models[0].state_dict(),
                    "model2": models[1].state_dict(),
                    "model3": models[2].state_dict(),
                    "weights": weights.cpu(),
                    "acc": acc,
                    "epoch": self.now_epoch,
                    "total_epoch": self.args.epoch
                }, model_path)
            else:
                torch.save({
                    "model1": models[0].state_dict(),
                    "model2": models[1].state_dict(),
                    "model3": None,
                    "weights": weights.cpu(),
                    "acc": acc,
                    "epoch": self.now_epoch,
                    "total_epoch": self.args.epoch
                }, model_path)
        logging.info("SYSTEM INFO : target acc : {}".format(acc))
        # ---- END ----
        target_key = "target_acc"  # 目标域的准确率列表
        target_best_key = "target_best"
        self.recorder.pull(target_key, acc)
        self.recorder.val2(target_best_key, acc)
