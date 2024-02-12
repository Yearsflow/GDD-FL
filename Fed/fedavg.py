from utils.common_utils import get_dataloader, DatasetSplit, get_network
import torch
import os
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import copy
import random
import torch.nn.functional as F
from networks import ResNet18
import math

class FedAvg(object):

    def __init__(self, args, appr_args, logger):
        self.args = args
        self.logger = logger
        self.appr_args = appr_args

    # function that processing the special arguments of current method
    @staticmethod
    def extra_parser(extra_args):
        parser = argparse.ArgumentParser()

        return parser.parse_args(extra_args)

    def partition(self, train_ds, val_ds):

        n_train = len(train_ds)
        n_val = len(val_ds)
        party2dataidx = {
            'train': {},
            'val': {}
        }

        if self.args.partition == 'homo' or self.args.partition == 'iid':
            for cur in ['train', 'val']:
                if cur == 'train':
                    idxs = np.random.permutation(n_train)
                else:
                    idxs = np.random.permutation(n_val)
                batch_idxs = np.array_split(idxs, self.args.C)
                party2dataidx[cur] = {i: batch_idxs[i] for i in range(self.args.C)}

        elif self.args.partition == 'noniid-labeldir' or self.args.partition == 'noniid':
            if self.args.dataset not in {'isic2020', 'EyePACS'}:
                for cur in ['train', 'val']:
                    min_size = 0
                    min_require_size = 10
                    K = 10
                    if self.args.dataset == 'cifar100':
                        K = 100
                    
                    if cur == 'train':
                        N = len(train_ds)
                        y = train_ds.targets
                    else:
                        N = len(val_ds)
                        y = val_ds.targets
                    y = np.array(y)

                    while min_size < min_require_size:
                        idx_batch = [[] for _ in range(self.args.C)]
                        for k in range(K):
                            idx_k = np.where(y == k)[0]
                            np.random.shuffle(idx_k)
                            proportions = np.random.dirichlet(np.repeat(self.args.alpha, self.args.C))
                            proportions = np.array([p * (len(idx_j) < N / self.args.C) for p, idx_j in zip(proportions, idx_batch)])
                            proportions = proportions / proportions.sum()
                            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                            min_size = min([len(idx_j) for idx_j in idx_batch])

                    for j in range(self.args.C):
                        np.random.shuffle(idx_batch[j])
                        party2dataidx[cur][j] = idx_batch[j]
            else:

                min_size = 0
                min_require_size = 10

                if self.args.dataset == 'isic2020':
                    K = 2
                elif self.args.dataset == 'EyePACS':
                    K = 5
                
                N = len(train_ds)
                y = train_ds.targets
                y = np.array(y)

                while True:
                    idx_batch = [[] for _ in range(self.args.C)]
                    for k in range(K):
                        idx_k = np.where(y == k)[0]
                        np.random.shuffle(idx_k)
                        proportions = np.random.dirichlet(np.repeat(self.args.alpha, self.args.C))
                        proportions = np.array([p * (len(idx_j) < N / self.args.C) for p, idx_j in zip(proportions, idx_batch)])
                        proportions = proportions / proportions.sum()
                        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                        min_size = min([len(idx_j) for idx_j in idx_batch])
                    flag = False
                    for idx_j in idx_batch:
                        n_per_class = [0 for _ in range(self.args.n_classes)]
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

                for j in range(self.args.C):
                    np.random.shuffle(idx_batch[j])
                    party2dataidx['train'][j] = idx_batch[j]

                idxs = np.random.permutation(n_val)
                batch_idxs = np.array_split(idxs, self.args.C)
                party2dataidx['val'] = {i: batch_idxs[i] for i in range(self.args.C)}
        
        return party2dataidx

    def run(self):

        self.logger.info('Partitioning data...')

        train_ds, val_ds, public_ds, test_ds, n_per_class = get_dataloader(self.args)
        self.party2dataidx = self.partition(train_ds, val_ds)

        self.party2datacounts = {}
        for c in range(self.args.C):
            n_per_client = [0 for _ in range(self.args.n_classes)]
            for idx in self.party2dataidx['train'][c]:
                n_per_client[train_ds.targets[idx]] += 1
            self.party2datacounts[c] = n_per_client
            
        print(self.party2datacounts)

        self.logger.info('Initialize nets...')
        
        local_nets = [get_network(self.args) for _ in range(self.args.C)]
        global_net = get_network(self.args)

        if self.args.device != 'cpu':
            global_net = nn.DataParallel(global_net)
            global_net.to(self.args.device)

        self.logger.info('Training begins...')
        
        for round in range(self.args.n_comm_round):
            self.logger.info('Communication Round: %d' % round)

            local_w = []

            party_list_this_round = random.sample([_ for _ in range(self.args.C)], 
                                                int(self.args.C * self.args.sample_fraction))
            party_list_this_round.sort()
    
            for c_id in party_list_this_round:
                self.logger.info('Client %d' % c_id)

                train_dl = DataLoader(DatasetSplit(train_ds, self.party2dataidx['train'][c_id]), num_workers=8, prefetch_factor=2*self.args.train_bs,
                                    batch_size=self.args.train_bs, shuffle=True, drop_last=False, pin_memory=True)
                val_dl = DataLoader(DatasetSplit(val_ds, self.party2dataidx['val'][c_id]), num_workers=8, prefetch_factor=2*self.args.test_bs,
                                    batch_size=self.args.test_bs, shuffle=False, pin_memory=True)
                self.logger.info('Train batches: %d' % len(train_dl))
                self.logger.info('Val batches: %d' % len(val_dl))

                if self.args.device != 'cpu':
                    local_nets[c_id] = nn.DataParallel(local_nets[c_id])
                    local_nets[c_id].to(self.args.device)

                # send global model to client
                local_nets[c_id].load_state_dict(global_net.state_dict())

                # local training
                class_weight = None
                if self.args.dataset == 'isic2020':
                    y = train_ds.targets
                    n_per_class = [0, 0]
                    for idx in self.party2dataidx['train'][c_id]:
                        n_per_class[y[idx]] += 1
                    class_weight = [sum(n_per_class) * 1.0 / (2 * n_per_class[0]),
                                    sum(n_per_class) * 1.0 / (2 * n_per_class[1])]
                elif self.args.dataset == 'EyePACS':
                    y = train_ds.targets
                    n_per_class = [0, 0, 0, 0, 0]
                    for idx in self.party2dataidx['train'][c_id]:
                        n_per_class[y[idx]] += 1
                    for i in range(len(n_per_class)):
                        if n_per_class[i] == 0:
                            n_per_class[i] = 1
                    class_weight = [sum(n_per_class) * 1.0 / (5 * n_per_class[0]),
                                    sum(n_per_class) * 1.0 / (5 * n_per_class[1]),
                                    sum(n_per_class) * 1.0 / (5 * n_per_class[2]),
                                    sum(n_per_class) * 1.0 / (5 * n_per_class[3]),
                                    sum(n_per_class) * 1.0 / (5 * n_per_class[4])]
                losses, accs, w = self.train(local_nets[c_id], train_dl, val_dl, class_weight)                    

                if (round + 1) % self.args.save_interval == 0 or round == 0:
                    torch.save(losses,
                        os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'losses_client{}_round{}_'.format(c_id, round)+self.args.log_file_name+'.pth'))
                    torch.save(accs,
                        os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'accs_client{}_round{}_'.format(c_id, round)+self.args.log_file_name+'.pth'))
                    torch.save(w,
                        os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'local_{}_client{}_round{}_'.format(self.args.model, c_id, round)+self.args.log_file_name+'.pth'))

                local_w.append(w)

            # global aggregation
            global_w = self.FedAvg(party_list_this_round, local_w)
            global_net.load_state_dict(global_w)      

            if (round + 1) % self.args.save_interval == 0 or round == 0:
                torch.save(global_w,
                    os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'global_{}_round{}_'.format(self.args.model, round)+self.args.log_file_name+'.pth'))

            test_dl = DataLoader(dataset=test_ds, batch_size=self.args.test_bs, shuffle=False,
                                num_workers=8, pin_memory=True, prefetch_factor=2*self.args.test_bs)

            if self.args.dataset not in {'isic2020', 'EyePACS'}:
                test_acc = self.eval(global_net, test_dl)
                self.logger.info('>>> Global Model Test Accuracy: %f' % test_acc)
            else:
                test_auc = self.eval(global_net, test_dl)
                self.logger.info('>>> Global Model Test AUC: %f' % test_auc)

    def train(self, net, train_dl, val_dl, class_weight=None):

        optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-5)
        
        if class_weight is not None:
            criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(class_weight)).float()).to(self.args.device)
        else:
            criterion = nn.CrossEntropyLoss()
        lf = lambda x: ((1 + math.cos(x * math.pi / self.args.epochs)) / 2) * (1 - self.args.lrf) + self.args.lrf
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        scaler = torch.cuda.amp.grad_scaler.GradScaler()

        train_losses, train_accs, train_aucs = [], [], []
        val_losses, val_accs, val_aucs = [], [], []

        best_val_acc, best_val_auc, best_model = 0, 0, None

        for epoch in range(self.args.epochs):
            epoch_loss_collector = []
            total, correct = 0, 0

            prob, targets = [], []
            net.train()
            
            for batch_idx, (x, target) in enumerate(train_dl):
                
                x = x.to(self.args.device, non_blocking=True)
                target = target.long().to(self.args.device, non_blocking=True)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast_mode.autocast():
                    out = net(x)
                    loss = criterion(out, target)

                total += x.data.size()[0]
                _, pred_label = torch.max(out.data, 1)
                if self.args.dataset == 'isic2020':
                    for i in range(len(out)):
                        prob.append(F.softmax(out[i], dim=0).cpu().tolist()[1])
                elif self.args.dataset == 'EyePACS':
                    for i in range(len(out)):
                        prob.append(F.softmax(out[i].double(), dim=0).cpu().tolist())
                targets += target.cpu().tolist()
                correct += (pred_label == target.data).sum().item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss_collector.append(loss.item())

            scheduler.step()
            
            epoch_train_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_train_acc = correct / float(total)
            if self.args.dataset == 'isic2020':
                epoch_train_auc = roc_auc_score(np.array(targets), np.array(prob))
                train_aucs.append(epoch_train_auc)
            elif self.args.dataset == 'EyePACS':
                epoch_train_auc = roc_auc_score(np.array(targets), np.array(prob), multi_class='ovo', labels=[0, 1, 2, 3, 4])
                train_aucs.append(epoch_train_auc)

            train_losses.append(epoch_train_loss)
            train_accs.append(epoch_train_acc)
            
            epoch_loss_collector = []
            total, correct = 0, 0
            
            prob, targets = [], []
            net.eval()

            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(val_dl):
                    
                    x = x.to(self.args.device, non_blocking=True)
                    target = target.long().to(self.args.device, non_blocking=True)
                    with torch.cuda.amp.autocast_mode.autocast():
                        out = net(x)
                        loss = criterion(out, target)
                    
                    _, pred_label = torch.max(out.data, 1)
                    if self.args.dataset == 'isic2020':
                        for i in range(len(out)):
                            prob.append(F.softmax(out[i], dim=0).cpu().tolist()[1])
                    elif self.args.dataset == 'EyePACS':
                        for i in range(len(out)):
                            prob.append(F.softmax(out[i].double(), dim=0).cpu().tolist())
                    targets += target.cpu().tolist()

                    epoch_loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

            epoch_val_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_val_acc = correct / float(total)
            if self.args.dataset == 'isic2020':
                epoch_val_auc = roc_auc_score(np.array(targets), np.array(prob))
                val_aucs.append(epoch_val_auc)
            elif self.args.dataset == 'EyePACS':
                epoch_val_auc = roc_auc_score(np.array(targets), np.array(prob), multi_class='ovo', labels=[0, 1, 2, 3, 4])
                val_aucs.append(epoch_val_auc)

            current_w = copy.deepcopy(net.state_dict())

            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)     

            if self.args.dataset in {'isic2020', 'EyePACS'}:
                if epoch_val_auc >= best_val_auc:
                    best_val_auc = epoch_val_auc
                    best_model = current_w
            else:
                if epoch_val_acc >= best_val_acc:
                    best_val_acc = epoch_val_acc
                    best_model = current_w
            
            if self.args.dataset in {'isic2020', 'EyePACS'}:
                self.logger.info('Epoch: %d Train Loss: %f Train AUC: %f Val Loss: %f Val AUC: %f' % (epoch, epoch_train_loss, epoch_train_auc, epoch_val_loss, epoch_val_auc))
            else:
                self.logger.info('Epoch: %d Train Loss: %f Train Acc: %f Val Loss: %f Val Acc: %f' % (epoch, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc))

        return (train_losses, val_losses), (train_accs, val_accs), best_model
    
    def eval(self, net, test_dl):

        total, correct = 0, 0
        prob, targets = [], []     

        net.eval()

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_dl):
                x = x.to(self.args.device, non_blocking=True)
                target = target.long().to(self.args.device, non_blocking=True)
                with torch.cuda.amp.autocast_mode.autocast():
                    out = net(x)

                if self.args.dataset == 'isic2020':
                    for i in range(len(out)):
                        prob.append(F.softmax(out[i], dim=0).cpu().tolist()[1])
                elif self.args.dataset == 'EyePACS':
                    for i in range(len(out)):
                        prob.append(F.softmax(out[i].double(), dim=0).cpu().tolist())
                targets += target.cpu().tolist()

                _, pred_label = torch.max(out.data, 1)
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

        test_acc = correct / float(total)
        if self.args.dataset == 'isic2020':
            test_auc = roc_auc_score(np.array(targets), np.array(prob))
        elif self.args.dataset == 'EyePACS':
            test_auc = roc_auc_score(np.array(targets), np.array(prob), multi_class='ovo', labels=[0, 1, 2, 3, 4])

        if self.args.dataset not in {'isic2020', 'EyePACS'}:
            return test_acc
        else:
            return test_auc
        
    def FedAvg(self, party_list_this_round, local_w):
        total_data_points = sum([len(self.party2dataidx['train'][c]) for c in party_list_this_round])
        fed_avg_freqs = [len(self.party2dataidx['train'][c]) / total_data_points for c in party_list_this_round]

        global_w = copy.deepcopy(local_w[0])
        for w_id, w in enumerate(local_w):
            if w_id == 0:
                for key in w:
                    global_w[key] = w[key] * fed_avg_freqs[w_id]
            else:
                for key in w:
                    global_w[key] += w[key] * fed_avg_freqs[w_id]
        
        return global_w