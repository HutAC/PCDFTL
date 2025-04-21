import torch
import pickle
import torch.optim as optim
import os
import warnings
from .transformer import awgn, shift_signal_padding_zero, time_stretch
import random
import torch.nn.functional as F
from sklearn.cluster import KMeans
from collections import Counter


def preprocess_features(data):
    """Preprocess an array of features using PyTorch.
    Args:
        data (torch.Tensor N x ndim): features to preprocess

    Returns:
        torch.Tensor of size N x pca: data PCA-reduced, whitened and L2-normalized
    """
    mean = data.mean(dim=0)
    centered_data = data - mean  # 去均值
    # L2 normalization
    centered_data = F.normalize(centered_data, dim=1)
    return centered_data


def check_dir():
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./checkPoint"):
        os.makedirs("./checkPoint")


def cos_sm(data):
    if len(data) == 1:
        return data[0]
    sm = torch.matmul(data, data.T)
    sm = sm.fill_diagonal_(0)
    sm = torch.sum(sm, dim=1)
    sm = sm/(len(data)-1)
    data = data * sm.unsqueeze(1)
    p = torch.sum(data, dim=0)/len(data)
    return p


# 有标签的原型获取 均值原型
def getFedByLabel(data_batch, label_batch, way="mean"):
    fed_dict = {}
    if len(data_batch) != len(label_batch):
        warnings.warn("data_batch size is not same as label_batch size !")
        return None
    for i in range(len(data_batch)):
        if label_batch[i] not in list(fed_dict.keys()):
            fed_dict[int(label_batch[i].item())] = []
        fed_dict[int(label_batch[i].item())].append(data_batch[i])
    labels = list(fed_dict.keys())
    for label in labels:
        fed_dict[label] = torch.stack(fed_dict[label], dim=0)
        if torch.isnan(fed_dict[label]).any():
            warnings.warn("client update warning: has nan")
            exit(0)
        if way == "mean":
            fed_dict[label] = torch.mean(fed_dict[label], dim=0)
        elif way == "cos":
            fed_dict[label] = cos_sm(fed_dict[label])
    return fed_dict


def getFedUnLabeled(data_batch, k, device):
    data_batch = preprocess_features(data_batch)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(data_batch)
    cluster_centers = kmeans.cluster_centers_
    labels = torch.tensor(kmeans.labels_).to(device).long()
    cluster_centers = torch.tensor(cluster_centers).to(device).float()
    return cluster_centers, labels

def toLabelFed(fed_dicts, unlabeled_fed):
    votes = {i: [] for i in range(len(unlabeled_fed))}
    for fed_dict in fed_dicts:
        label_fed = []
        for key in list(fed_dict.keys()):
            label_fed.append(fed_dict[key])
        label_fed = torch.stack(label_fed, dim=0)
        sm = torch.matmul(unlabeled_fed, label_fed.T)
        print(sm)
        max_ = torch.argmax(sm, dim=1)
        for i, v in enumerate(max_):
            votes[i].append(int(v.item()))
    result = {}
    print(votes)
    for i in range(len(unlabeled_fed)):
        counter = Counter(votes[i])
        most, count = counter.most_common(1)[0]
        result[most] = unlabeled_fed[i]
    return result


def accuracy(outputs, labels):
    """
    Compute the accuracy
    outputs, labels: (tensor)
    return: (float) accuracy in [0, 100]
    """
    pre = torch.max(outputs.cpu(), 1)[1].numpy()
    y = labels.data.cpu().numpy()
    acc = ((pre == y).sum() / len(y)) * 100
    return acc


# ===== Test the Model =====
def tester(featurenet, classifier, dataloader):
    featurenet.eval()
    classifier.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    correct_num, total_num = 0, 0
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = x_batch.permute(0, 2, 1)
            x_batch = x_batch.unsqueeze(-1)
            logtis_batch = featurenet(x_batch)
            output_batch = classifier(logtis_batch)
            pre = torch.max(output_batch.cpu(), 1)[1].numpy()
            y = y_batch.cpu().numpy()
            correct_num += (pre == y).sum()
            total_num += len(y)
        ac = (correct_num / total_num) * 100.0
        return ac

def optimizer(args, parameter_list):
    # define optimizer
    if args.optimizer == "sgd":
        optimizer = optim.SGD(parameter_list, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(parameter_list, lr=args.lr, weight_decay=1e-4)
    else:
        raise Exception("optimizer not implement")

    # Define the learning rate decay
    if args.lr_scheduler == 'step':
        steps = [int(step) for step in args.steps.split(',')]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=args.gamma)
    elif args.lr_scheduler == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    elif args.lr_scheduler == 'stepLR':
        steps = int(args.steps.split(",")[0])
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steps, args.gamma)
    elif args.lr_scheduler == 'cos':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, 0)
    elif args.lr_scheduler == 'fix':
        lr_scheduler = None
    else:
        raise Exception("lr schedule not implement")

    return optimizer, lr_scheduler


def glabel(data: torch.tensor, labels: torch.tensor):
    label_set = set(labels)
    data_dict = {}
    for k in label_set:
        key = int(k.item())
        if key not in data_dict.keys():
            data_dict[key] = []

    for index, label in enumerate(labels):
        data_dict[int(label.item())].append(data[index])

    for key in data_dict.keys():
        data_dict[key] = torch.stack(data_dict[key], dim=0)
    return data_dict

def signal_batch_transformer(batch):
    result = [[], [], []]
    for b in batch:
        awgnSnr = random.uniform(5, 15)
        timeFactor = random.uniform(0.8, 1.2)
        shiftNum = random.randint(32, 64)
        x = b[:, 0]
        y = b[:, 1]
        z = b[:, 2]
        awgnX = awgn(x, awgnSnr)
        awgnY = awgn(y, awgnSnr)
        awgnZ = awgn(z, awgnSnr)
        awgnSignal = torch.stack((awgnX, awgnY, awgnZ), dim=1)
        result[0].append(awgnSignal)
        shitfX = shift_signal_padding_zero(x, shiftNum)
        shitfY = shift_signal_padding_zero(y, shiftNum)
        shitfZ = shift_signal_padding_zero(z, shiftNum)
        shitfSignal = torch.stack((shitfX, shitfY, shitfZ), dim=1)
        result[1].append(shitfSignal)
        timeX = time_stretch(x, timeFactor)
        timeY = time_stretch(y, timeFactor)
        timeZ = time_stretch(z, timeFactor)
        timeSignal = torch.stack((timeX, timeY, timeZ), dim=1)
        if timeFactor > 1:
            timeSignal = timeSignal[:len(x), :]
        else:
            padding = abs(len(x)-len(timeX))
            timeX = F.pad(timeX, (0, padding), 'constant', 0)
            timeY = F.pad(timeY, (0, padding), 'constant', 0)
            timeZ = F.pad(timeZ, (0, padding), 'constant', 0)
            timeSignal = torch.stack((timeX, timeY, timeZ), dim=1)
        result[2].append(timeSignal)
    for i in range(len(result)):
        result[i] = torch.stack(result[i], dim=0)
    new_batch = torch.cat(result, dim=0)
    return new_batch


def lock_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def unlock_grad(model):
    for param in model.parameters():
        param.requires_grad = True


if __name__ == "__main__":
    signal = torch.randn((16, 1024, 3))
    signal_batch_transformer(signal)
    # X = np.array([[1, 2], [1, 4], [1, 0],
    #               [4, 2], [4, 4], [4, 0],
    #               [2, 3], [2, 5], [2, 1]])
    # X = torch.Tensor(X)
    # getFedUnLabeled(X, k=3)