import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import random

import argparse
import logging
import os
import copy
import datetime
import math

import xgboost as xgb
import pandas as pd


from model import *
from networks import ResNet18
# from datasets import MNIST_truncated, SVHN_custom, CustomTensorDataset, CelebA_custom, ImageFolder_custom, PneumoniaDataset, ImageFolder_public
from dataset_utils import MNIST_truncated, CIFAR10_truncated, CIFAR100_truncated, DatasetSplit, TensorDataset, ImageFolder_custom
from common_utils import seed_everything, get_dataset_info
from trees import *
from torchvision import datasets

libsvm_datasets = {
    "a9a": "binary_cls",
    "cod-rna": "binary_cls"
}


n_workers = 8

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ResNet18', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid', help='how to partition the dataset on local workers')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='how many epochs will be trained in a training process')
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--n_teacher_each_partition', type=int, default=1,
                        help='number of local nets in a partitioning of a party')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='which type of communication strategy is going to be used: fedavg/fedkt/fedprox/simenb')
    parser.add_argument('--comm_round', type=int, default=1,
                            help='number of communication rounds')
    parser.add_argument('--trials', type=int, default=1, help="Number of trials for each run")
    parser.add_argument('--init_seed', type=int, default=42, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=True, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--max_tree_depth', type=int, default=6, help='Max tree depth for the tree model')
    parser.add_argument('--n_ensemble_models', type=int, default=10, help="Number of the models in the final ensemble")

    parser.add_argument('--train_local_student', type=int, default=1, help="whether use PATE to train local student models before aggregation")
    parser.add_argument('--auxiliary_data_portion', type=float, default=0.5, help="the portion of test data that is used as the auxiliary data for PATE")
    parser.add_argument('--stu_epochs', type=int, default=100, help='Number of epochs for the student model')
    parser.add_argument('--with_unlabeled', type=int, default=1, help='Whether there is public unlabeled data')
    parser.add_argument('--stu_lr', type=float, default=0.001, help='The learning rate for the student model')
    parser.add_argument('--is_local_split', type=int, default=1, help='Whether split the local data for local model training')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--ensemble_method', type=str, default='max_vote', help='Choice: max_vote or averaging')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--n_partition', type=int, default=1, help='The partition times of each party')
    parser.add_argument('--gamma', type=float, default=None, help='The parameter for differential privacy')
    parser.add_argument('--privacy_analysis_file_name', type=str, default=None, help='The file path to save the information for privacy analysis')
    parser.add_argument('--n_stu_trees', type=int, default=100, help='The number of trees in a student model')
    parser.add_argument('--optimizer', type=str, default='sgd', help='sgd or adam optimizer')
    parser.add_argument('--local_training_epochs', type=int, default=None, help='the number of epochs for the local trainig alg')
    parser.add_argument('--dp_level', type=int, default=0, help='1 represents add dp on the server side. 2 represents add dp on the party side')
    parser.add_argument('--query_portion', type=float, default=0.5, help='how many queries are used to train the final model')
    parser.add_argument('--local_query_portion', type=float, default=0.5, help='how many queries are used to train the student models')
    parser.add_argument('--filter_query', type=int, default=0, help='Whether to filter the query or not')
    parser.add_argument('--max_z', type=int, default=1, help='the maximum partition that may be influenced when changing a single record')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--fedkt_seed', type=int, default=42, help='the seed before run fedkt')
    parser.add_argument('--pub_datadir', type=str, default=None, help='the path to the public data')
    parser.add_argument('--prob_threshold', type=float, default=None, help='a threshold to filter the votes')
    parser.add_argument('--min_require', type=int, default=None, help='require that the minimum number of samples of each class is at least min_require')
    parser.add_argument('--prob_threshold_apply', type=int, default=0,
                        help='0 means no apply, 1 means apply only at server part, 2 means apply only at party part, 3 means apply at both parts')
    parser.add_argument('--apply_consistency', type=int, default=1, help='the votes of the party will only be counted if they are the same if set to 1')
    parser.add_argument('--save_global_model', type=int, default=0, help='whether save the global model or not')
    parser.add_argument('--final_stu_epochs', type=int, default=100, help='the number of epochs to train the final student model')
    parser.add_argument('--init_std', type=float, default=-1, help='the stdv for the initialization of the weights, -1 for norm initialization')
    parser.add_argument('--std_place', type=int, default=0, help='1 for std in teacher model, 2 add student model')
    parser.add_argument('--retrain_local_epoch', type=int, default=10, help='the local epoch in fedavg/fedprox after fedkt')
    parser.add_argument('--n_final_stu_trees', type=int, default=100, help='the number of trees of the final model')
    parser.add_argument('--npartyseed', type=str, default=None, help='nparty-seed')
    parser.add_argument('--new_scaffold', type=int, default=0, help='whether use new scaffold')
    parser.add_argument('--train_bs', type=int, default=64, help='train batch size')
    parser.add_argument('--test_bs', type=int, default=64, help='test batch size')
    args = parser.parse_args()
    return args


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_mnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_svhn_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    svhn_train_ds = SVHN_custom(datadir, train=True, download=True, transform=transform)
    svhn_test_ds = SVHN_custom(datadir, train=False, download=True, transform=transform)

    X_train, y_train = svhn_train_ds.data, svhn_train_ds.target
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.target

    # X_train = X_train.data.numpy()
    # y_train = y_train.data.numpy()
    # X_test = X_test.data.numpy()
    # y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def partition(args, train_ds, val_ds):

    n_train = len(train_ds)
    n_val = len(val_ds)
    party2dataidx = {
        'train': {},
        'val': {}
    }

    args.C = args.n_parties
    args.alpha = args.beta

    if args.partition == 'homo' or args.partition == 'iid':
        for cur in ['train', 'val']:
            if cur == 'train':
                idxs = np.random.permutation(n_train)
            else:
                idxs = np.random.permutation(n_val)
            batch_idxs = np.array_split(idxs, args.C)
            party2dataidx[cur] = {i: batch_idxs[i] for i in range(args.C)}

    elif args.partition == 'noniid-labeldir' or args.partition == 'noniid':
        if args.dataset not in {'isic2020', 'EyePACS'}:
            for cur in ['train', 'val']:
                min_size = 0
                min_require_size = 10
                K = 10
                if args.dataset == 'cifar100':
                    K = 100
                
                if cur == 'train':
                    N = len(train_ds)
                    y = train_ds.targets
                else:
                    N = len(val_ds)
                    y = val_ds.targets
                y = np.array(y)

                while min_size < min_require_size:
                    idx_batch = [[] for _ in range(args.C)]
                    for k in range(K):
                        idx_k = np.where(y == k)[0]
                        np.random.shuffle(idx_k)
                        proportions = np.random.dirichlet(np.repeat(args.alpha, args.C))
                        proportions = np.array([p * (len(idx_j) < N / args.C) for p, idx_j in zip(proportions, idx_batch)])
                        proportions = proportions / proportions.sum()
                        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                        min_size = min([len(idx_j) for idx_j in idx_batch])

                for j in range(args.C):
                    np.random.shuffle(idx_batch[j])
                    party2dataidx[cur][j] = idx_batch[j]
        else:

            min_size = 0
            min_require_size = 10

            if args.dataset == 'isic2020':
                K = 2
            elif args.dataset == 'EyePACS':
                K = 5
            
            N = len(train_ds)
            y = train_ds.targets
            y = np.array(y)

            while True:
                idx_batch = [[] for _ in range(args.C)]
                for k in range(K):
                    idx_k = np.where(y == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(args.alpha, args.C))
                    proportions = np.array([p * (len(idx_j) < N / args.C) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
                flag = False
                for idx_j in idx_batch:
                    n_per_class = [0 for _ in range(args.n_classes)]
                    for idx in idx_j:
                        n_per_class[y[idx]] += 1
                    classes = 0
                    for num in n_per_class:
                        if num != 0:
                            classes += 1
                    if classes <= 1:
                        flag = True
                        break
                if not flag:
                    break

            for j in range(args.C):
                np.random.shuffle(idx_batch[j])
                party2dataidx['train'][j] = idx_batch[j]

            idxs = np.random.permutation(n_val)
            batch_idxs = np.array_split(idxs, args.C)
            party2dataidx['val'] = {i: batch_idxs[i] for i in range(args.C)}
    
    return party2dataidx


def init_nets(net_configs, dropout_p, n_parties, args, n_teacher_each_partition = 1, stdv=None):

    n_total_nets = n_parties * n_teacher_each_partition
    nets = {net_i: None for net_i in range(n_total_nets)}

    for net_i in range(n_total_nets):
        if args.model == "mlp":
            input_size = net_configs[0]
            output_size = net_configs[-1]
            hidden_sizes = net_configs[1:-1]
            net = FcNet(input_size, hidden_sizes, output_size, stdv, dropout_p)
        # elif args.model == "vgg":
        #     net = vgg11()
        elif args.model == "simple-cnn":
            if args.dataset in ("svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == "mnist":
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == 'celeba' or args.dataset == 'xray':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
        elif args.model == "vgg-9":
            if args.dataset in ("mnist"):
                net = ModerateCNNMNIST()
            elif args.dataset in ("svhn"):
                # print("in moderate cnn")
                net = ModerateCNN()
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2)
        # elif args.model == "resnet":
        #     net = ResNet50()
        # elif args.model == "vgg16":
        #     net = vgg16()
        elif args.model == 'lr':
            if args.dataset == 'a9a':
                net = LogisticRegression(123,2)
        elif args.model == 'ResNet18':
            if args.dataset == 'mnist':
                net = ResNet18(args, channel=1, num_classes=10)
            elif args.dataset == 'cifar10':
                net = ResNet18(args, channel=3, num_classes=10)
            elif args.dataset == 'isic2020':
                net = ResNet18(args, channel=3, num_classes=2)
                num_ftrs = net.classifier.in_features
                net.classifier = nn.Linear(num_ftrs * 4 * 4, args.n_classes)
            elif args.dataset == 'EyePACS':
                net = ResNet18(args, channel=3, num_classes=5)
                num_ftrs = net.classifier.in_features
                net.classifier = nn.Linear(num_ftrs * 4 * 4, args.n_classes)
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def init_weights(m):
  if type(m)==nn.Linear or type(m)==nn.Conv2d:
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)

def get_trainable_parameters(net):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    # print("net.parameter.data:", list(net.parameters()))
    paramlist=list(trainable)
    N=0
    for params in paramlist:
        N+=params.numel()
        # print("params.data:", params.data)
    X=torch.empty(N,dtype=torch.float64)
    X.fill_(0.0)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            X[offset:offset+numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset+=numel
    # print("get trainable x:", X)
    return X

def get_all_parameters(net):
    'return trainable parameter values as a vector (only the first parameter set)'
    # print("net.parameter.data:", list(net.parameters()))
    paramlist=list(net.parameters())
    N=0
    for params in paramlist:
        N+=params.numel()
        # print("params.data:", params.data)
    X=torch.empty(N,dtype=torch.float64)
    X.fill_(0.0)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            X[offset:offset+numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset+=numel
    # print("get trainable x:", X)
    return X

def put_trainable_parameters(net,X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset+numel].data.view_as(params.data))
        offset+=numel

def put_all_parameters(net,X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    paramlist=list(net.parameters())
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset+numel].data.view_as(params.data))
        offset+=numel


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target, _) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            out = model(x)
            _, pred_label = torch.max(out.data, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)

def compute_AUC(model, dataloader, get_confusion_matrix=False, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target, _) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            out = model(x)
            _, pred_label = torch.max(out.data, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)

def prepare_weight_matrix(n_classes, weights: dict):
    weights_list = {}

    for net_i, cls_cnts in weights.items():
        cls = np.array(list(cls_cnts.keys()))
        cnts = np.array(list(cls_cnts.values()))
        weights_list[net_i] = np.array([0] * n_classes, dtype=np.float32)
        weights_list[net_i][cls] = cnts
        weights_list[net_i] = torch.from_numpy(weights_list[net_i]).view(1, -1)

    return weights_list


def prepare_uniform_weights(n_classes, net_cnt, fill_val=1):
    weights_list = {}

    for net_i in range(net_cnt):
        temp = np.array([fill_val] * n_classes, dtype=np.float32)
        weights_list[net_i] = torch.from_numpy(temp).view(1, -1)

    return weights_list


def prepare_sanity_weights(n_classes, net_cnt):
    return prepare_uniform_weights(n_classes, net_cnt, fill_val=0)


def normalize_weights(weights):
    Z = np.array([])
    eps = 1e-6
    weights_norm = {}

    for _, weight in weights.items():
        if len(Z) == 0:
            Z = weight.data.numpy()
        else:
            Z = Z + weight.data.numpy()

    for mi, weight in weights.items():
        weights_norm[mi] = weight / torch.from_numpy(Z + eps)

    return weights_norm


def get_weighted_average_pred(models: list, weights: dict, x, device="cpu"):
    out_weighted = None

    # Compute the predictions
    for model_i, model in enumerate(models):
        #logger.info("Model: {}".format(next(model.parameters()).device))
        #logger.info("data device: {}".format(x.device))
        out = F.softmax(model(x), dim=-1)  # (N, C)
        # print("model(x):", model(x))
        # print("out:", out)

        weight = weights[model_i].to(device)


        if out_weighted is None:
            weight = weight.to(device)
            out_weighted = (out * weight)
        else:
            out_weighted += (out * weight)

    return out_weighted


def get_pred_votes(models, x, threshold=None, device="cpu"):
    # print("input x:", x)
    # Compute the predictions
    votes=torch.LongTensor([]).to(device)
    for model_i, model in enumerate(models):
        #logger.info("Model: {}".format(next(model.parameters()).device))
        #logger.info("data device: {}".format(x.device))
        out = F.softmax(model(x), dim=-1)  # (N, C)
        pred_probs, pred_label = torch.max(out,1)
        if threshold is not None:
            # pred_probs.to("cpu")
            # pred_label.to("cpu")
            for index, prob in enumerate(pred_probs):
                if prob < threshold:
                    pred_label[index] = -1
            # pred_label.to(device)
        votes=torch.cat((votes, pred_label),dim=0)

    return votes


def compute_ensemble_accuracy(models: list, dataloader, n_classes, ensemble_method="max_vote", train_cls_counts=None,
                              uniform_weights=False, sanity_weights=False, device="cpu"):

    correct, total = 0, 0
    true_labels_list, pred_labels_list = np.array([]), np.array([])

    was_training = [False]*len(models)
    for i, model in enumerate(models):
        if model.training:
            was_training[i] = True
            model.eval()
    if ensemble_method == "averaging":
        if uniform_weights is True:
            weights_list = prepare_uniform_weights(n_classes, len(models))
        elif sanity_weights is True:
            weights_list = prepare_sanity_weights(n_classes, len(models))
        else:
            weights_list = prepare_weight_matrix(n_classes, train_cls_counts)

        weights_norm = normalize_weights(weights_list)

    with torch.no_grad():
        for batch_idx, (x, target, _) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            target = target.long()
            if ensemble_method == "averaging":
                out = get_weighted_average_pred(models, weights_norm, x, device=device)
                _, pred_label = torch.max(out, 1)
            elif ensemble_method == "max_vote":
                votes = get_pred_votes(models, x, device=device)
                pred_label, _ = torch.mode(votes.view(-1, x.data.size()[0]), dim=0)


            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    #logger.info(correct, total)

    conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    for i, model in enumerate(models):
        if was_training[i]:
            model.train()

    return correct / float(total), conf_matrix

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)
    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target, _) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            #for adam l2 reg
            # l2_reg = torch.zeros(1)
            # l2_reg.requires_grad = True

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        # logger.info('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)

        if epoch % 10 == 0:
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
            train_acc = compute_accuracy(net, train_dataloader, device=device)
            test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

            logger.info('>> Training accuracy: %f' % train_acc)
            logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    return train_acc, test_acc



def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu, model, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target, _) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            #for adam l2 reg
            # l2_reg = torch.zeros(1)
            # l2_reg.requires_grad = True

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            #for fedprox
            fed_prox_reg = 0.0
            # fed_prox_reg += np.linalg.norm([i - j for i, j in zip(global_weight_collector, get_trainable_parameters(net).tolist())], ord=2)
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
            loss += fed_prox_reg


            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        # logger.info('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        if epoch % 10 == 0:
            train_acc = compute_accuracy(net, train_dataloader, device=device)
            test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

            logger.info('>> Training accuracy: %f' % train_acc)
            logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_scaffold(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, args, server_c, client_c, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # mu = 0.001
    global_collector = list(global_net.to(device).parameters())
    server_c_collector = list(server_c.to(device).parameters())
    client_c_collector = list(client_c.to(device).parameters())
    client_c_delta = copy.deepcopy(client_c_collector)

    c_global_para = get_all_parameters(server_c)
    c_local_para = get_all_parameters(client_c)

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target, _) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            #for adam l2 reg
            # l2_reg = torch.zeros(1)
            # l2_reg.requires_grad = True

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            loss.backward()
            for param_index, param in enumerate(net.parameters()):
                param.grad += server_c_collector[param_index] - client_c_collector[param_index]
            optimizer.step()

            # net_para = get_all_parameters(net)
            # net_para = net_para - args.lr * (c_global_para - c_local_para)
            # put_all_parameters(net, net_para)


            # for param_index, param in enumerate(net.parameters()):
            #     r_grad = param.requires_grad
            #     param.requires_grad = False
            #     param -= args.lr*(server_c_collector[param_index] - client_c_collector[param_index])
            #     param.requires_grad = r_grad
            cnt += 1
            epoch_loss_collector.append(loss.item())

        # logger.info('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        if epoch % 10 == 0:
            train_acc = compute_accuracy(net, train_dataloader, device=device)
            test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

            logger.info('>> Training accuracy: %f' % train_acc)
            logger.info('>> Test accuracy: %f' % test_acc)

    for param_index, param in enumerate(net.parameters()):
        client_c_delta[param_index] = (global_collector[param_index] - param) / (
                    args.epochs * len(train_dataloader) * lr) - server_c_collector[param_index]
        client_c_collector[param_index] += client_c_delta[param_index]
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    logger.info(' ** Training complete **')
    return train_acc, test_acc, client_c_delta


def save_model(model, model_index, args):
    logger.info("saving model-{}".format(model_index))
    with open(os.path.join(args.logdir, args.log_file_name) + ".model" + str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


def load_model(model, model_index, rank=0, device="cpu"):
    #
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model


def local_train_net(nets, args, net_dataidx_map, X_train = None, y_train = None, X_test = None, y_test = None, remain_test_dl = None, local_split=False, retrain_epoch=None, device="cpu"):
    # save local dataset
    # local_datasets = []
    n_teacher_each_partition = args.n_teacher_each_partition
    avg_acc = 0.0
    if local_split:
        split_datasets = []
        for party_id in range(args.n_parties):
            np.random.shuffle(net_dataidx_map[party_id])
            split_datasets.append(np.array_split(net_dataidx_map[party_id], args.n_teacher_each_partition))

    for net_id, net in nets.items():
        if not local_split:
            dataidxs = net_dataidx_map[net_id//n_teacher_each_partition]
        else:
            dataidxs = list(split_datasets[net_id//n_teacher_each_partition][net_id%n_teacher_each_partition])

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        if args.dataset in libsvm_datasets:
            pass
            # party_id = net_id // n_teacher_each_partition
            # train_ds_local = CustomTensorDataset(torch.tensor(X_train[net_dataidx_map[party_id]].toarray(), dtype=torch.float32),
            #                                      torch.tensor(y_train[net_dataidx_map[party_id]], dtype=torch.long))
            # public_ds = CustomTensorDataset(torch.tensor(X_test[:public_data_size].toarray(), dtype=torch.float32),
            #                                torch.tensor(y_test[:public_data_size], dtype=torch.long))
            # remain_test_ds = CustomTensorDataset(torch.tensor(X_test[public_data_size:].toarray(), dtype=torch.float32),
            #                                     torch.tensor(y_test[public_data_size:], dtype=torch.long))
            # train_dl_local = data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
            # remain_test_dl = data.DataLoader(dataset=remain_test_ds, batch_size=32, shuffle=False)
        else:
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)

        # local_datasets.append((train_dl_local, test_dl_local))

        # switch to global test set here

        # if remain_test_dl is not None:
        #     test_dl_global = remain_test_dl
        if args.alg == 'local_training':
            n_epoch = args.local_training_epochs
        else:
            n_epoch = args.epochs

        if retrain_epoch is not None:
            n_epoch = retrain_epoch

        trainacc, testacc = train_net(net_id, net, train_dl_local, remain_test_dl, n_epoch, args.lr, args.optimizer, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_fedprox(nets, global_model, args, net_dataidx_map, X_train = None, y_train = None, X_test = None, y_test = None, remain_test_dl = None, local_split=False, retrain_epoch=None, device="cpu"):
    # save local dataset
    # local_datasets = []
    n_teacher_each_partition = args.n_teacher_each_partition
    avg_acc = 0.0
    if local_split:
        split_datasets = []
        for party_id in range(args.n_parties):
            np.random.shuffle(net_dataidx_map[party_id])
            split_datasets.append(np.array_split(net_dataidx_map[party_id], args.n_teacher_each_partition))

    for net_id, net in nets.items():
        if not local_split:
            dataidxs = net_dataidx_map[net_id//n_teacher_each_partition]
        else:
            dataidxs = list(split_datasets[net_id//n_teacher_each_partition][net_id%n_teacher_each_partition])

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        if args.dataset in libsvm_datasets:
            party_id = net_id//n_teacher_each_partition
            train_ds_local = CustomTensorDataset(torch.tensor(X_train[net_dataidx_map[party_id]].toarray(), dtype=torch.float32),
                                                 torch.tensor(y_train[net_dataidx_map[party_id]], dtype=torch.long))
            public_ds = CustomTensorDataset(torch.tensor(X_test[:public_data_size].toarray(), dtype=torch.float32),
                                           torch.tensor(y_test[:public_data_size], dtype=torch.long))
            remain_test_ds = CustomTensorDataset(torch.tensor(X_test[public_data_size:].toarray(), dtype=torch.float32),
                                                torch.tensor(y_test[public_data_size:], dtype=torch.long))
            train_dl_local = data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True, num_workers=n_workers)
            remain_test_dl = data.DataLoader(dataset=remain_test_ds, batch_size=32, shuffle=False)
        else:
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)


        if args.alg == 'local_training':
            n_epoch = args.local_training_epochs
        else:
            n_epoch = args.epochs

        if retrain_epoch is not None:
            n_epoch = retrain_epoch
        trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, remain_test_dl, n_epoch, args.lr, args.optimizer, args.mu, args.model, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

def local_train_net_scaffold(nets, global_model, args, net_dataidx_map, X_train = None, y_train = None, X_test = None, y_test = None, server_c=None, clients_c=None, remain_test_dl = None, local_split=False, device="cpu"):
    n_teacher_each_partition = args.n_teacher_each_partition
    avg_acc = 0.0
    if local_split:
        split_datasets = []
        for party_id in range(args.n_parties):
            np.random.shuffle(net_dataidx_map[party_id])
            split_datasets.append(np.array_split(net_dataidx_map[party_id], args.n_teacher_each_partition))
    server_c_collector = list(server_c.to(device).parameters())
    new_server_c_collector = copy.deepcopy(server_c_collector)
    for net_id, net in nets.items():
        if not local_split:
            dataidxs = net_dataidx_map[net_id // n_teacher_each_partition]
        else:
            dataidxs = list(split_datasets[net_id // n_teacher_each_partition][net_id % n_teacher_each_partition])

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        if args.dataset in libsvm_datasets:
            party_id = net_id // n_teacher_each_partition
            train_ds_local = CustomTensorDataset(
                torch.tensor(X_train[net_dataidx_map[party_id]].toarray(), dtype=torch.float32),
                torch.tensor(y_train[net_dataidx_map[party_id]], dtype=torch.long))
            public_ds = CustomTensorDataset(torch.tensor(X_test[:public_data_size].toarray(), dtype=torch.float32),
                                            torch.tensor(y_test[:public_data_size], dtype=torch.long))
            remain_test_ds = CustomTensorDataset(torch.tensor(X_test[public_data_size:].toarray(), dtype=torch.float32),
                                                 torch.tensor(y_test[public_data_size:], dtype=torch.long))
            train_dl_local = data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True,
                                             num_workers=n_workers)
            remain_test_dl = data.DataLoader(dataset=remain_test_ds, batch_size=32, shuffle=False)
        else:
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs)
            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)

        if args.alg == 'local_training':
            n_epoch = args.local_training_epochs
        else:
            n_epoch = args.epochs

        trainacc, testacc, c_delta = train_net_scaffold(net_id, net, global_model, train_dl_local, remain_test_dl, n_epoch,
                                              args.lr, args.optimizer, args, server_c, clients_c[net_id], device=device)
        if args.new_scaffold:
            for param_index, param in enumerate(server_c.parameters()):
                new_server_c_collector[param_index] += c_delta[param_index] / args.n_parties
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    if args.new_scaffold:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_on_a_party(nets, args, net_dataidx_map, party_id, train_ds, val_ds, remain_test_dl = None, local_split=0, device="cpu"):
    # save local dataset
    # local_datasets = []
    n_teacher_each_partition = args.n_teacher_each_partition
    if local_split:
        train_split_datasets, val_split_datasets = [], []
        np.random.shuffle(net_dataidx_map['train'][party_id])
        np.random.shuffle(net_dataidx_map['val'][party_id])
        train_split_datasets = np.array_split(net_dataidx_map['train'][party_id], args.n_teacher_each_partition)
        val_split_datasets = np.array_split(net_dataidx_map['val'][party_id], args.n_teacher_each_partition)

    for net_id, net in nets.items():
        if not local_split:
            dataidxs = net_dataidx_map[party_id]
        else:
            train_dataidxs = list(train_split_datasets[net_id])
            val_dataidxs = list(val_split_datasets[net_id])

        logger.info("Training network %s. n_training: %d n_val: %d" % (str(net_id), len(train_dataidxs), len(val_dataidxs)))
        # move the model to cuda device:
        net.to(device)

        if args.dataset in libsvm_datasets:
            pass
            # train_ds_local = CustomTensorDataset(torch.tensor(X_train[net_dataidx_map[party_id]].toarray(), dtype=torch.float32),
            #                                      torch.tensor(y_train[net_dataidx_map[party_id]], dtype=torch.long))
            # public_ds = CustomTensorDataset(torch.tensor(X_test[:public_data_size].toarray(), dtype=torch.float32),
            #                                torch.tensor(y_test[:public_data_size], dtype=torch.long))
            # remain_test_ds = CustomTensorDataset(torch.tensor(X_test[public_data_size:].toarray(), dtype=torch.float32),
            #                                     torch.tensor(y_test[public_data_size:], dtype=torch.long))
            # train_dl_local = data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
            # test_dl_global = data.DataLoader(dataset=remain_test_ds, batch_size=32, shuffle=False)
        else:
            train_dl_local = data.DataLoader(DatasetSplit(train_ds, train_dataidxs), num_workers=8, prefetch_factor=2*args.train_bs,
                                    batch_size=args.train_bs, shuffle=True, drop_last=False, pin_memory=True)
            val_dl_local = data.DataLoader(DatasetSplit(val_ds, val_dataidxs), num_workers=8, prefetch_factor=2*args.test_bs,
                                    batch_size=args.test_bs, shuffle=False, pin_memory=True)
            test_dl_global = remain_test_dl
            # train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
            # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)

        # local_datasets.append((train_dl_local, test_dl_local))

        # switch to global test set here
        if remain_test_dl is not None:
            test_dl_global = remain_test_dl
        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl_global, args.epochs, args.lr, args.optimizer, device=device)
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)

    nets_list = list(nets.values())
    return nets_list


def central_train_net_on_a_party(nets, args, X_train = None, y_train = None, X_test = None, y_test = None, remain_test_dl = None, local_split=0, device="cpu"):
    # save local dataset
    # local_datasets = []
    n_teacher_each_partition = args.n_teacher_each_partition

    dataidx_arr = np.arange(len(y_train))
    np.random.shuffle(dataidx_arr)
    # partition the local data to n_local_models parts
    dataidx = np.array_split(dataidx_arr, n_teacher_each_partition)

    for net_id, net in nets.items():

        # logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        if args.dataset in libsvm_datasets:
            train_ds_local = CustomTensorDataset(torch.tensor(X_train[net_dataidx_map[party_id]].toarray(), dtype=torch.float32),
                                                 torch.tensor(y_train[net_dataidx_map[party_id]], dtype=torch.long))
            public_ds = CustomTensorDataset(torch.tensor(X_test[:public_data_size].toarray(), dtype=torch.float32),
                                           torch.tensor(y_test[:public_data_size], dtype=torch.long))
            remain_test_ds = CustomTensorDataset(torch.tensor(X_test[public_data_size:].toarray(), dtype=torch.float32),
                                                torch.tensor(y_test[public_data_size:], dtype=torch.long))
            train_dl_local = data.DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True)
            test_dl_global = data.DataLoader(dataset=remain_test_ds, batch_size=32, shuffle=False)
        else:
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidx[net_id])
            train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)

        # local_datasets.append((train_dl_local, test_dl_local))

        # switch to global test set here
        if remain_test_dl is not None:
            test_dl_global = remain_test_dl
        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl_global, args.epochs, args.lr, args.optimizer, device=device)
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)

    nets_list = list(nets.values())
    return nets_list

def get_dataloader(args):

    num_per_class = None
    
    if args.dataset == 'mnist':

        n_train, n_val, n_public = 50000, 5000, 5000
        idxs = [_ for _ in range(60000)]
        random.shuffle(idxs)
        train_idxs = idxs[: n_train]
        val_idxs = idxs[n_train: n_train + n_val]
        public_idxs = idxs[n_train + n_val: n_train + n_val + n_public]

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])

        if args.approach == 'fedgan':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])

        train_ds = MNIST_truncated(args.datadir, dataidxs=train_idxs, train=True, transform=transform_train, download=True)
        val_ds = MNIST_truncated(args.datadir, dataidxs=val_idxs, train=True, transform=transform_test, download=True)
        public_ds = MNIST_truncated(args.datadir, dataidxs=public_idxs, train=True, transform=transform_test, download=True)
        test_ds = MNIST_truncated(args.datadir, train=False, transform=transform_test, download=True)
        
    elif args.dataset == 'cifar10':

        n_train, n_val, n_public = 40000, 5000, 5000
        idxs = [_ for _ in range(50000)]
        random.shuffle(idxs)
        train_idxs = idxs[: n_train]
        val_idxs = idxs[n_train: n_train + n_val]
        public_idxs = idxs[n_train + n_val: n_train + n_val + n_public]

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], 
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        ])

        if args.approach == 'fedgan':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], 
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        ])

        train_ds = CIFAR10_truncated(args.datadir, dataidxs=train_idxs, train=True, transform=transform_train, download=True)
        val_ds = CIFAR10_truncated(args.datadir, dataidxs=val_idxs, train=True, transform=transform_test, download=True)
        public_ds = CIFAR10_truncated(args.datadir, dataidxs=public_idxs, train=True, transform=transform_test, download=True)
        test_ds = CIFAR10_truncated(args.datadir, train=False, transform=transform_test, download=True)
        
    elif args.dataset == 'cifar100':

        n_train, n_val, n_public = 40000, 5000, 5000
        idxs = [_ for _ in range(50000)]
        random.shuffle(idxs)
        train_idxs = idxs[: n_train]
        val_idxs = idxs[n_train: n_train + n_val]
        public_idxs = idxs[n_train + n_val: n_train + n_val + n_public]

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])

        train_ds = CIFAR100_truncated(args.datadir, dataidxs=train_idxs, train=True, transform=transform_train, download=True)
        val_ds = CIFAR100_truncated(args.datadir, dataidxs=val_idxs, train=True, transform=transform_test, download=True)
        test_ds = CIFAR100_truncated(args.datadir, train=False, transform=transform_test, download=True)
        
    elif args.dataset == 'isic2020':

        n_train = [19381, 351]
        n_val, n_public = [3326, 58], [3326, 58]
        n_test = [6509, 117]

        transform_train = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if args.approach == 'fedgan':
            transform_train = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        transform_test = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_ds =ImageFolder_custom(root=os.path.join(args.datadir, 'train'), transform=transform_train)
        val_ds = ImageFolder_custom(root=os.path.join(args.datadir, 'val'), transform=transform_test)
        public_ds = ImageFolder_custom(root=os.path.join(args.datadir, 'public'), transform=transform_test)
        test_ds = ImageFolder_custom(root=os.path.join(args.datadir, 'test'), transform=transform_test)

        num_per_class = {
            'train': n_train,
            'val': n_val,
            'public': n_public,
            'test': n_test
        }

    elif args.dataset == 'EyePACS':

        n_train = [23229, 2199, 4763, 786, 638]
        n_public, n_val = [2581, 244, 529, 87, 70], [8130, 720, 1579, 237, 240]
        n_test = [31403, 3042, 6282, 977, 966]

        transform_train = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if args.approach == 'fedgan':
            transform_train = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        transform_test = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_ds = ImageFolder_custom(root=os.path.join(args.datadir, 'train'), transform=transform_train)
        public_ds = ImageFolder_custom(root=os.path.join(args.datadir, 'public'), transform=transform_test)
        val_ds = ImageFolder_custom(root=os.path.join(args.datadir, 'val'), transform=transform_test)
        test_ds = ImageFolder_custom(root=os.path.join(args.datadir, 'test'), transform=transform_test)

        num_per_class = {
            'train': n_train,
            'val': n_val,
            'public': n_public,
            'test': n_test
        }

    return train_ds, val_ds, public_ds, test_ds, num_per_class


def get_prediction_labels(models, n_classes, dataloader, args, gamma=None, method="max_vote", train_cls_counts=None,
                          uniform_weights=True, sanity_weights=False, is_subset=0, is_final_student = False, device="cpu"):
    # correct, total = 0, 0
    # true_labels_list = []
    pred_labels_list = []

    was_training = [False]*len(models)
    for i, model in enumerate(models):
        if model.training:
            was_training[i] = True
            model.eval()

    if method == "averaging":
        if uniform_weights is True:
            weights_list = prepare_uniform_weights(n_classes, len(models))
        elif sanity_weights is True:
            weights_list = prepare_sanity_weights(n_classes, len(models))
        else:
            weights_list = prepare_weight_matrix(n_classes, train_cls_counts)

        weights_norm = normalize_weights(weights_list)


    vote_counts_save = np.empty((n_classes, 0), dtype=int)
    correct, total = 0, 0
    top2_counts_differ_one = 0
    with torch.no_grad():
        for batch_idx, (x, target, index) in enumerate(dataloader):
            x = x.to(device)
            # target = target.to(device)
            # target = target.long()

            if method == "averaging":
                out = get_weighted_average_pred(models, weights_norm, x, device=device)
                _, pred_label = torch.max(out, 1)
            elif method == "max_vote":
                votes = get_pred_votes(models, x, args.prob_threshold, device=device)
                votes_view = votes.view(-1, x.data.size()[0])

                vote_counts_real = torch.LongTensor([]).to(device)
                for class_id in range(n_classes):
                    if (args.apply_consistency == 2) or (is_final_student and args.apply_consistency):
                        vote_count_real = torch.zeros(x.data.size()[0], dtype=torch.long, device=device)
                        for pid in range(args.n_parties):
                            votes_view_perparty = votes_view[pid*args.n_partition : (pid+1)*args.n_partition]
                            vote_count_real_party = (votes_view_perparty == class_id).sum(dim=0, dtype=torch.long)
                            # vote_count_real_party = vote_count_real_party * ((vote_count_real_party>=math.ceil(args.n_partition*args.nvote_threshold)).long())
                            vote_count_real += vote_count_real_party
                    else:
                        vote_count_real = (votes_view == class_id).sum(dim=0)
                    vote_counts_real = torch.cat((vote_counts_real, vote_count_real), dim=0)
                # print("vote_counts_real:", vote_counts_real)
                # print("vote counts view:", vote_counts_real.view(-1,x.data.size()[0]))
                vote_counts_save = np.append(vote_counts_save, vote_counts_real.view(-1,x.data.size()[0]).to("cpu").numpy(), axis=1)
                # print("vote_counts_save:", vote_counts_save)
                if gamma is None or gamma == 0:
                    # pred_label, _ = torch.mode(votes_view, dim=0)
                    _, pred_label = torch.max(vote_counts_real.view(-1, x.data.size()[0]), 0)
                else:
                    vote_counts = torch.FloatTensor([]).to(device)

                    for class_id in range(n_classes):
                        vote_count = (votes_view==class_id).sum(dim=0).float()
                        if args.apply_consistency and is_final_student:
                            for idx, vote in enumerate(vote_count):
                                if vote != args.n_partition:
                                    vote_count[idx] = 0
                        for i in range(len(vote_count)):
                            vote_count[i]+=np.random.laplace(loc=0.0, scale=float(1.0/gamma))
                        vote_counts=torch.cat((vote_counts,vote_count),dim=0)

                    # print("vote_counts:", vote_counts.to("cpu"))
                    # print("vote_counts view:", vote_counts.view(-1,x.data.size()[0]).to("cpu"))

                    _, pred_label=torch.max(vote_counts.view(-1,x.data.size()[0]),0)

            total += x.data.size()[0]
            # correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list.append(list(pred_label.numpy()))
                # true_labels_list.append(list(target.data.numpy()))

                if is_subset == 2:
                    dataloader.dataset.dataset.dataset.target[index] = torch.LongTensor(list(pred_label.numpy()))
                elif is_subset == 1:
                    dataloader.dataset.dataset.target[index]=torch.LongTensor(list(pred_label.numpy()))
                else:
                    dataloader.dataset.targets[index]=torch.LongTensor(list(pred_label.numpy()))

            else:
                pred_labels_list.append(list(pred_label.cpu().numpy()))
                # true_labels_list.append(list(target.data.cpu().numpy()))

                if is_subset == 2:
                    dataloader.dataset.dataset.dataset.target[index] = torch.LongTensor(list(pred_label.cpu().numpy()))
                elif is_subset == 1:
                    dataloader.dataset.dataset.target[index] = torch.LongTensor(list(pred_label.cpu().numpy()))
                else:
                    dataloader.dataset.targets[index] = torch.LongTensor(list(pred_label.cpu().numpy()))
            # print("target:", target)
            # target = torch.LongTensor(list(pred_label.numpy())).cpu()

    vote_counts_save = np.transpose(vote_counts_save)


    top1_class_counts = np.zeros(500)
    top2_class_counts = np.zeros(500)
    top_diff_counts = np.zeros(500)
    for row in vote_counts_save:
        top2_counts = row[np.argsort(row)[-2:]]
        if top2_counts[1] - top2_counts[0] <= 1:
            top2_counts_differ_one+=1
        top_diff_counts[top2_counts[1] - top2_counts[0]] += 1
        top1_class_counts[top2_counts[1]] += 1
        top2_class_counts[top2_counts[0]] += 1

    return pred_labels_list, top2_counts_differ_one, vote_counts_save


def train_a_student(tea_nets, public_dataloader, public_ds, remain_test_dataloader, stu_net, n_classes, args,
                    gamma=None, is_subset=0, is_final_student=False, filter_query=0,device = 'cpu'):
    if args.pub_datadir is not None:
        is_subset = 0
    public_labels, top2_counts_differ_one, vote_counts_save = get_prediction_labels(tea_nets, n_classes, public_dataloader, args, gamma=gamma,
                                          method=args.ensemble_method, is_subset=is_subset, is_final_student= is_final_student, device=device)
    if filter_query:
        confident_query_idx = []
        for idx, row in enumerate(vote_counts_save):
            top2_counts = row[np.argsort(row)[-2:]]
            if top2_counts[1] - top2_counts[0] > 2:
                confident_query_idx.append(idx)
        print("len confident query idx:", len(confident_query_idx))
        logger.info("len confident query idx: %d" % len(confident_query_idx))
        # local_query_ds = data.Subset(public_ds, confident_query_idx)
        public_dataloader = data.DataLoader(dataset=public_ds, batch_size=32, sampler=data.SubsetRandomSampler(confident_query_idx), num_workers=n_workers)



    logger.info('len public_labels: %d' % len(public_labels))
    logger.info('Training student network')
    logger.info('n_public: %d' % len(public_ds))
    stu_net.to(device)

    train_acc = compute_accuracy(stu_net, public_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    # logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, stu_net.parameters()), lr=args.stu_lr, weight_decay=args.reg)
    elif args.optimizer == 'adam_ams':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, stu_net.parameters()), lr=args.stu_lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, stu_net.parameters()), lr=args.stu_lr, momentum=0.9, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0

    if is_final_student:
        n_epoch = args.final_stu_epochs
    else:
        n_epoch = args.stu_epochs

    for epoch in range(n_epoch):
        epoch_loss_collector = []
        for batch_idx, (x, target, _) in enumerate(public_dataloader):
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = stu_net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

            # test_acc = compute_accuracy(stu_net, remain_test_dataloader, device=device)
            # test_acc, conf_matrix = compute_accuracy(stu_net, test_dataloader, get_confusion_matrix=True, device=device)

            # logger.info('>> Test accuracy in epoch %d: %f' % (epoch, test_acc))

        # logger.info('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        if epoch % 10 == 0:

            train_acc = compute_accuracy(stu_net, public_dataloader, device=device)
            test_acc = compute_accuracy(stu_net, remain_test_dataloader, device=device)

            logger.info('>> Training accuracy: %f  Test accuracy: %f' % (train_acc, test_acc))


    logger.info(' ** Training complete **')
    return train_acc, top2_counts_differ_one, vote_counts_save



if __name__ == '__main__':
    
    args = get_args()
    args.approach = 'FedKT'
    args.n_classes, args.channel, args.im_size, \
    args.mean, args.std = get_dataset_info(args.dataset)
    seed_everything(args.init_seed)
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s-%d-%d' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"),args.init_seed, args.trials)
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),

        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(device)

    test_accs=[]
    for n_exp in range(args.trials):
        # seed = n_exp + args.init_seed
        seed = args.init_seed
        logger.info("#" * 100)
        logger.info("Executing Trial %d with seed %d" % (n_exp, seed))
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger.info("Partitioning data")
        # if args.alg == "pate":
        #     args.partition = 'homo'

        # X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        #     args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta, min_require=args.min_require)

        train_ds, val_ds, public_ds, test_ds, n_per_class = get_dataloader(args)
        party2dataidx = partition(args, train_ds, val_ds)

        if args.dataset in {'mnist', 'cifar10'}:
            n_classes = 10
        elif args.dataset == 'isic2020':
            n_classes = 2
        elif args.dataset == 'EyePACS':
            n_classes = 5
        # n_classes = len(np.unique(y_train))

        if args.dataset not in libsvm_datasets:

            # train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
            #                                                                                   args.datadir,
            #                                                                                   args.batch_size,
            #                                                                                   32)
            train_ds_global = train_ds
            test_ds_global = test_ds

            print("len train_ds_global:", len(train_ds_global))

            # public_data_size = int(len(test_ds_global) * args.auxiliary_data_portion)
            # query_data_size = int(len(test_ds_global) * args.query_portion)
            # local_query_data_size = int(len(test_ds_global) * args.local_query_portion)

            public_data_size = len(public_ds)
            query_data_size = len(public_ds)
            local_query_data_size = len(public_ds)
            remain_data_size = len(test_ds_global)
            # unquery_size = len(test_ds_global) - query_data_size
            # local_unquery_size = len(test_ds_global) - local_query_data_size
            if args.dataset !='xray':
                # public_ds, remain_test_ds = data.random_split(test_ds_global,
                #                                               [public_data_size, remain_data_size])
                remain_test_ds = test_ds_global
                public_dl = data.DataLoader(dataset=public_ds, batch_size=64, shuffle=True, num_workers=n_workers,
                                            pin_memory=True, prefetch_factor=2*64)

                # query_dl = data.DataLoader(dataset=query_ds, batch_size=32, shuffle=False)
                # local_query_dl = data.DataLoader(dataset=local_query_ds, batch_size=32, shuffle=False)
                remain_test_dl = data.DataLoader(dataset=remain_test_ds, batch_size=64, shuffle=False, num_workers=n_workers,
                                                 pin_memory=True, prefetch_factor=2*64)

                query_dl = data.DataLoader(dataset=public_ds, batch_size=64, shuffle=True, num_workers=n_workers,
                                           pin_memory=True, prefetch_factor=2*64)
                local_query_dl = data.DataLoader(dataset=public_ds, batch_size=64, shuffle=True, num_workers=n_workers,
                                                 pin_memory=True, prefetch_factor=2*64)

                # query_dl = data.DataLoader(dataset=public_ds, batch_size=32,
                #                            sampler=data.SubsetRandomSampler(list(range(query_data_size))), num_workers=n_workers)
                # local_query_dl = data.DataLoader(dataset=public_ds, batch_size=32,
                #                                  sampler=data.SubsetRandomSampler(list(range(local_query_data_size))), num_workers=n_workers)

        if args.alg == 'fedkt' or args.alg == 'fedkt_fedavg' or args.alg=='fedkt_fedprox' or args.alg=='simenb':
            if args.fedkt_seed is not None:
                np.random.seed(args.fedkt_seed)
                torch.manual_seed(args.fedkt_seed)
            
            if args.model == 'ResNet18':
                logger.info("Initializing nets")
                if args.n_teacher_each_partition == 1:
                    args.is_local_split = 0
                    args.train_local_student = 0
                logger.info("Training nets")

                tea_nets = []
                n_parti_top2_differ_one = np.zeros(args.n_parties)
                n_instances_portion = np.zeros(args.n_parties)
                vote_counts_parties = []

                prob_threshold = args.prob_threshold
                if args.prob_threshold_apply != 2 and args.prob_threshold_apply != 3:
                    args.prob_threshold = None
                for party_id in range(args.n_parties):

                    #start training student models
                    for i in range(args.n_partition):
                        is_subset_temp = 0
                        stu_public_ds = public_ds
                        stu_query_dl = query_dl
                        stu_local_query_dl = local_query_dl
                        filter_query = args.filter_query
                        if i < args.max_z:
                            filter_query = 0
                        if args.std_place > 0:
                            init_std = args.init_std
                        else:
                            init_std = None
                        nets, _, _ = init_nets(args.net_config, args.dropout_p, 1, args,
                                               args.n_teacher_each_partition, init_std)
                        nets_list = local_train_net_on_a_party(nets, args, party2dataidx, party_id, train_ds, val_ds, remain_test_dl=remain_test_dl, local_split=args.is_local_split,
                                                    device=device)
                        # nets_list_partition.append(nets_list)

                        if args.train_local_student:
                            if not args.with_unlabeled:
                                print("need public unlabeled data!")
                                exit(1)

                            logger.info("in party %d" % party_id)
                            if args.std_place > 1:
                                init_std = args.init_std
                            else:
                                init_std = None
                            stu_nets, _, _ = init_nets(args.net_config, args.dropout_p, 1, args, 1, init_std)
                            stu_net = stu_nets[0]
                            if args.dp_level == 2:
                                gamma = args.gamma
                                _, top2_counts_differ_one, vote_counts_in_a_party = train_a_student(nets_list,
                                                                                                    stu_local_query_dl,
                                                                                                    stu_public_ds,
                                                                                                    remain_test_dl,
                                                                                                    stu_net, n_classes,
                                                                                                    args, gamma=gamma,
                                                                                                    is_subset=is_subset_temp,
                                                                                                    filter_query=filter_query,
                                                                                                    device=device)
                            else:
                                gamma = 0
                                _, top2_counts_differ_one, vote_counts_in_a_party = train_a_student(nets_list, stu_local_query_dl, stu_public_ds,
                                                remain_test_dl, stu_net, n_classes, args, gamma=gamma, is_subset=is_subset_temp, filter_query=filter_query,device=device)
                            vote_counts_parties = np.append(vote_counts_parties, vote_counts_in_a_party)
                            if top2_counts_differ_one != 0:
                                n_parti_top2_differ_one[party_id] += 1


                            local_stu_test_acc, conf_mat = compute_accuracy(stu_net, remain_test_dl, get_confusion_matrix=True, device=device)
                            logger.info("local_stu_test_acc: %f" % local_stu_test_acc)

                            tea_nets.append(stu_net)
                        else:
                            for net in nets_list:
                                tea_nets.append(net)

                    n_instances_portion[party_id] = len(party2dataidx['train'][party_id]) / len(train_ds_global)
                if args.prob_threshold_apply == 1 or args.prob_threshold_apply == 3:
                    args.prob_threshold = prob_threshold
                else:
                    args.prob_threshold = None
                # print("portion sum:", np.sum(n_instances_portion))
                # if args.alg == 'simenb':
                #     simple_ensemble_acc, _ = compute_ensemble_accuracy(tea_nets, remain_test_dl, n_classes, ensemble_method=args.ensemble_method, uniform_weights=True, device=device)
                #     logger.info("simple ensemble acc:%f" % simple_ensemble_acc)
                #     exit(0)
                if args.with_unlabeled:
                    global_stu_nets, _, _=init_nets(args.net_config, args.dropout_p, 1, args, 1)
                    global_stu_net = global_stu_nets[0]
                    if args.dp_level == 1:
                        gamma = args.gamma
                        _, _, vote_counts = train_a_student(tea_nets, query_dl, public_ds, remain_test_dl, global_stu_net,
                                                            n_classes, args, gamma=args.gamma,
                                                            is_subset=1, is_final_student=True, filter_query=0,device=device)
                    else:
                        gamma = 0
                        _, _, vote_counts = train_a_student(tea_nets, query_dl, public_ds, remain_test_dl, global_stu_net,
                                                            n_classes, args, gamma=args.gamma,
                                                            is_subset=0, is_final_student=True, filter_query=0,device=device)
                    # can change to local data to train the student

                    global_stu_test_acc, conf_mat = compute_accuracy(global_stu_net, remain_test_dl, get_confusion_matrix=True, device=device)
                    test_acc = global_stu_test_acc
                    logger.info("global_stu_test_acc: %f"% global_stu_test_acc)
                else:
                    print("not supported yet")


                if args.privacy_analysis_file_name is not None:
                    file_path = os.path.join(args.logdir, args.privacy_analysis_file_name + "-%d" % n_exp)
                else:
                    file_path = os.path.join(args.logdir, log_path + "-%d" % n_exp)
                file_path1 = file_path + '-dp0'
                file_path2 = file_path + '-dp1'
                np.savez(file_path1, n_instances_portion, n_parti_top2_differ_one, vote_counts)
                if args.train_local_student:
                    np.savez(file_path2, n_instances_portion, n_parti_top2_differ_one, vote_counts_parties.reshape(-1,10))
                global_model = global_stu_net
                if args.save_global_model:
                    save_model(global_model, 0, args)
            