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
from .fedavg import FedAvg
from utils.common_utils import get_dataloader, DatasetSplit, get_network
import torch.nn.functional as F
import math

class FedProto(FedAvg):

    def __init__(self, args, appr_args, logger):
        super(FedProto, self).__init__(args, appr_args, logger)

    def run(self):

        self.logger.info('Partitioning data...')

        train_ds, val_ds, public_ds, test_ds, num_per_class = get_dataloader(self.args)
        self.party2dataidx = self.partition(train_ds, val_ds)

        self.logger.info('Initialize nets...')

        local_nets = [get_network(self.args) for _ in range(self.args.C)]

        self.logger.info('Training begins...')

        global_proto = []

        for round in range(self.args.n_comm_round):
            
            self.logger.info('Communication Round: %d' % round)

            local_protos = {}
            party_list_this_round = random.sample([_ for _ in range(self.args.C)], 
                                                int(self.args.C * self.args.sample_fraction))
            party_list_this_round.sort()

            for c in party_list_this_round:

                self.logger.info('Client %d' % c)

                train_dl = DataLoader(DatasetSplit(train_ds, self.party2dataidx['train'][c]), num_workers=8, prefetch_factor=2*self.args.train_bs,
                                    batch_size=self.args.train_bs, shuffle=True, drop_last=False, pin_memory=True)
                val_dl = DataLoader(DatasetSplit(val_ds, self.party2dataidx['val'][c]), num_workers=8, prefetch_factor=2*self.args.test_bs,
                                    batch_size=self.args.test_bs, shuffle=False, pin_memory=True)
            
                self.logger.info('Train batches: %d' % len(train_dl))
                self.logger.info('Val batches: %d' % len(val_dl))

                if self.args.device != 'cpu':
                    local_nets[c] = nn.DataParallel(local_nets[c])
                    local_nets[c].to(self.args.device)

                # local training
                class_weight = None
                if self.args.dataset == 'isic2020':
                    y = train_ds.targets
                    n_per_class = [0, 0]
                    for idx in self.party2dataidx['train'][c]:
                        n_per_class[y[idx]] += 1
                    class_weight = [sum(n_per_class) * 1.0 / (2 * n_per_class[0]),
                                    sum(n_per_class) * 1.0 / (2 * n_per_class[1])]
                elif self.args.dataset == 'EyePACS':
                    y = train_ds.targets
                    n_per_class = [0, 0, 0, 0, 0]
                    for idx in self.party2dataidx['train'][c]:
                        n_per_class[y[idx]] += 1
                    for i in range(len(n_per_class)):
                        if n_per_class[i] == 0:
                            n_per_class[i] = 1
                    class_weight = [sum(n_per_class) * 1.0 / (5 * n_per_class[0]),
                                    sum(n_per_class) * 1.0 / (5 * n_per_class[1]),
                                    sum(n_per_class) * 1.0 / (5 * n_per_class[2]),
                                    sum(n_per_class) * 1.0 / (5 * n_per_class[3]),
                                    sum(n_per_class) * 1.0 / (5 * n_per_class[4])]
                losses, accs, w, proto = self.train(local_nets[c], train_dl, val_dl, class_weight)
                agg_protos = self.agg_func(proto)

                if (round + 1) % self.args.save_interval == 0 or round == 0:
                    torch.save(losses,
                        os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'losses_client{}_round{}_'.format(c, round)+self.args.log_file_name+'.pth'))
                    torch.save(accs,
                        os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'accs_client{}_round{}_'.format(c, round)+self.args.log_file_name+'.pth'))
                    torch.save(w,
                        os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'local_{}_client{}_round{}_'.format(self.args.model, c, round)+self.args.log_file_name+'.pth'))
                    torch.save(proto,
                        os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'local_proto_client{}_round{}_'.format(c, round)+self.args.log_file_name+'.pth'))

                local_protos[c] = agg_protos

            # global aggregation
            global_proto = self.proto_aggregation(local_protos)

            test_dl = DataLoader(dataset=test_ds, batch_size=self.args.test_bs, shuffle=False,
                                num_workers=8, pin_memory=True, prefetch_factor=2*self.args.test_bs)

            local_model_wo_metrics, local_model_w_metrics = [], []

            for c in range(self.args.C):
                test_wo_metrics, test_w_metrics = self.inference(local_nets[c], test_dl, global_proto)
                if self.args.dataset not in {'isic2020', 'EyePACS'}:
                    self.logger.info('>>> Client %d Local Model w/o Global Proto Test Acc: %f' % (c, test_wo_metrics))
                    self.logger.info('>>> Client %d Local Model w Global Proto Test Acc: %f' % (c, test_w_metrics))
                else:
                    self.logger.info('>>> Client %d Local Model w/o Global Proto Test AUC: %f' % (c, test_wo_metrics))
                    self.logger.info('>>> Client %d Local Model w Global Proto Test AUC: %f' % (c, test_w_metrics))
                local_model_wo_metrics.append(test_wo_metrics)
                local_model_w_metrics.append(test_w_metrics)

            if self.args.dataset not in {'isic2020', 'EyePACS'}:
                self.logger.info('>>>>> Average w/o Test Acc: %f' % (np.mean(np.array(local_model_wo_metrics))))
                self.logger.info('>>>>> Average w Test Acc: %f' % (np.mean(np.array(local_model_w_metrics))))
            else:
                self.logger.info('>>>>> Average w/o Test AUC: %f' % (np.mean(np.array(local_model_wo_metrics))))
                self.logger.info('>>>>> Average w Test AUC: %f' % (np.mean(np.array(local_model_w_metrics))))          
            
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
            agg_protos_label = {}
            epoch_loss_collector = []
            total, correct = 0, 0

            net.train()

            prob, targets = [], []
            
            for batch_idx, (x, target) in enumerate(train_dl):
                x = x.to(self.args.device, non_blocking=True)
                target = target.long().to(self.args.device, non_blocking=True)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast_mode.autocast():
                    out, protos = net(x)
                    loss = criterion(out, target)
                
                protos = protos.to('cpu')
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

                for i in range(len(target)):
                    if target[i].item() in agg_protos_label:
                        agg_protos_label[target[i].item()].append(protos[i,:])
                    else:
                        agg_protos_label[target[i].item()] = [protos[i,:]]

                epoch_loss_collector.append(loss.item())
            
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
            net.eval()
            
            prob, targets = [], []

            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(val_dl):
                    x = x.to(self.args.device, non_blocking=True)
                    target = target.long().to(self.args.device, non_blocking=True)
                    with torch.cuda.amp.autocast_mode.autocast():
                        out, protos = net(x)
                        loss = criterion(out, target)
                    
                    protos = protos.to('cpu')
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

            current_w = copy.deepcopy(net.state_dict())

            epoch_val_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_val_acc = correct / float(total)
            if self.args.dataset == 'isic2020':
                epoch_val_auc = roc_auc_score(np.array(targets), np.array(prob))
                val_aucs.append(epoch_val_auc)
            elif self.args.dataset == 'EyePACS':
                epoch_val_auc = roc_auc_score(np.array(targets), np.array(prob), multi_class='ovo', labels=[0, 1, 2, 3, 4])
                val_aucs.append(epoch_val_auc)

            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc) 

            scheduler.step()    

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

        return (train_losses, val_losses), (train_accs, val_accs), best_model, agg_protos_label
    
    def agg_func(self, protos):
        '''
        Returns the average of the weights.
        '''

        for [label, proto_list] in protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                protos[label] = proto / len(proto_list)
            else:
                protos[label] = proto_list[0]

        return protos
    
    def proto_aggregation(self, local_protos_list):
        agg_protos_label = dict()
        for idx in local_protos_list:
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = [proto / len(proto_list)]
            else:
                agg_protos_label[label] = [proto_list[0].data]

        return agg_protos_label
    
    def inference(self, net, test_dl, global_proto=[]):

        if self.args.dataset in ['mnist', 'cifar10']:
            n_classes = 10
        elif self.args.dataset == 'cifar100':
            n_classes = 100
        elif self.args.dataset == 'isic2020':
            n_classes = 2
        elif self.args.dataset == 'EyePACS':
            n_classes = 5
        else:
            raise NotImplementedError('Dataset Not Supported')
        
        total, correct = 0, 0
        prob, targets = [], []
        loss_mse = nn.MSELoss()
        scaler = torch.cuda.amp.grad_scaler.GradScaler()

        net.eval()

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_dl):
                x = x.to(self.args.device, non_blocking=True)
                target = target.long().to(self.args.device, non_blocking=True)
                with torch.cuda.amp.autocast_mode.autocast():
                    out, proto = net(x)
                proto = proto.to('cpu')

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

        test_acc_wo_proto = correct / float(total)
        if self.args.dataset == 'isic2020':
            test_auc_wo_proto = roc_auc_score(np.array(targets), np.array(prob))   
        elif self.args.dataset == 'EyePACS':
            test_auc_wo_proto = roc_auc_score(np.array(targets), np.array(prob), multi_class='ovo', labels=[0, 1, 2, 3, 4])    

        if global_proto != []:

            total, correct = 0, 0
            preds, targets = [], []

            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(test_dl):
                    x = x.to(self.args.device, non_blocking=True)
                    target = target.long().to(self.args.device, non_blocking=True)
                    with torch.cuda.amp.autocast_mode.autocast():
                        out, proto = net(x)
                    proto = proto.to('cpu')

                    # compute the dist between protos and global_protos
                    a_large_num = 100
                    dist = a_large_num * torch.ones(size=(x.shape[0], n_classes)).to(self.args.device)  # initialize a distance matrix
                    for i in range(x.shape[0]):
                        for j in range(n_classes):
                            if j in global_proto.keys():
                                d = loss_mse(proto[i, :], global_proto[j][0])
                                dist[i, j] = d

                    # prediction
                    _, pred_label = torch.min(dist, 1)
                    targets += target.cpu().tolist()
                    correct += (pred_label == target.data).sum().item()
                    pred_label = pred_label.cpu().tolist()
                    if self.args.dataset == 'EyePACS':
                        for i in range(len(pred_label)):
                            pred_score = [0 for _ in range(n_classes)]
                            pred_score[pred_label[i]] = 1
                            preds.append(pred_score)
                    elif self.args.dataset == 'isic2020':
                        preds += pred_label
                    total += x.data.size()[0]
            
            test_acc_w_proto = correct / float(total)
            if self.args.dataset == 'isic2020':
                test_auc_w_proto = roc_auc_score(np.array(targets), np.array(preds))
                return (test_auc_wo_proto, test_auc_w_proto)
            elif self.args.dataset == 'EyePACS':
                test_auc_w_proto = roc_auc_score(np.array(targets), np.array(preds), multi_class='ovo', labels=[0, 1, 2, 3, 4])
                return (test_auc_wo_proto, test_auc_w_proto)

            return (test_acc_wo_proto, test_acc_w_proto)
        
        if self.args.dataset == 'isic2020':
            return test_auc_wo_proto
        elif self.args.dataset == 'EyePACS':
            return test_auc_wo_proto
        return test_acc_wo_proto