from utils.common_utils import get_dataloader, get_network
from networks import ResNet18
import torch
import os
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader
import math

class Central(object):

    def __init__(self, args, appr_args, logger):
        self.args = args
        self.appr_args = appr_args
        self.logger = logger

    # function that processing the special arguments of current method
    @staticmethod
    def extra_parser(extra_args):
        parser = argparse.ArgumentParser()

        return parser.parse_args(extra_args)

    def run(self):

        train_ds, val_ds, public_ds, test_ds, num_per_class = get_dataloader(self.args)
        train_dl = DataLoader(train_ds, batch_size=self.args.train_bs, shuffle=True, pin_memory=True,
                              num_workers=8, drop_last=False, prefetch_factor=2*self.args.train_bs)
        val_dl = DataLoader(val_ds, batch_size=self.args.test_bs, shuffle=False, pin_memory=True,
                            num_workers=8, drop_last=False, prefetch_factor=2*self.args.test_bs)
        test_dl = DataLoader(test_ds, batch_size=self.args.test_bs, shuffle=False, pin_memory=True,
                            num_workers=8, drop_last=False, prefetch_factor=2*self.args.test_bs)
        
        net = get_network(self.args)
        if self.args.device != 'cpu':
            net = nn.DataParallel(net)
            net.to(self.args.device)

        if self.args.dataset not in {'isic2020', 'EyePACS'}:
            losses, accs, w = self.train(net, train_dl, val_dl)
        else:
            if self.args.dataset == 'isic2020':
                class_weight = [sum(num_per_class['train']) * 1.0 / (2 * num_per_class['train'][0]),
                                sum(num_per_class['train']) * 1.0 / (2 * num_per_class['train'][1])]
            else:
                class_weight = [sum(num_per_class['train']) * 1.0 / (5 * num_per_class['train'][0]),
                                sum(num_per_class['train']) * 1.0 / (5 * num_per_class['train'][1]),
                                sum(num_per_class['train']) * 1.0 / (5 * num_per_class['train'][2]),
                                sum(num_per_class['train']) * 1.0 / (5 * num_per_class['train'][3]),
                                sum(num_per_class['train']) * 1.0 / (5 * num_per_class['train'][4]),]
            losses, accs, w = self.train(net, train_dl, val_dl, class_weight)

        torch.save(losses,
            os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'losses_'+self.args.log_file_name+'.pth'))
        torch.save(accs,
            os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'accs_'+self.args.log_file_name+'.pth'))
        torch.save(w,
            os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'model_'+self.args.log_file_name+'.pth'))
        
        net.load_state_dict(w)

        if self.args.dataset not in {'isic2020', 'EyePACS'}:
            test_acc = self.eval(net, test_dl)
            self.logger.info('>>>>> Test Accuracy: %f' % test_acc)
        else:
            test_auc = self.eval(net, test_dl)
            self.logger.info('>>>>> Test AUC: %f' % test_auc)

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
            net.train()

            prob, targets = [], []
            
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
                else:
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
            
            train_losses.append(epoch_train_loss)
            train_accs.append(epoch_train_acc)
            
            if self.args.dataset == 'isic2020':
                epoch_train_auc = roc_auc_score(np.array(targets), np.array(prob))
                train_aucs.append(epoch_train_auc)
            elif self.args.dataset == 'EyePACS':
                epoch_train_auc = roc_auc_score(np.array(targets), np.array(prob), multi_class='ovo', labels=[0, 1, 2, 3, 4])
                train_aucs.append(epoch_train_auc)
            
            epoch_loss_collector = []
            total, correct = 0, 0
            net.eval()
            
            prob, targets = [], []

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
                    else:
                        for i in range(len(out)):
                            prob.append(F.softmax(out[i].double(), dim=0).cpu().tolist())
                    targets += target.cpu().tolist()

                    epoch_loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

            epoch_val_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_val_acc = correct / float(total)

            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)

            if self.args.dataset == 'isic2020':
                epoch_val_auc = roc_auc_score(np.array(targets), np.array(prob))
                val_aucs.append(epoch_val_auc)
                if epoch_val_auc >= best_val_auc:
                    best_val_auc = epoch_val_auc
                    best_model = copy.deepcopy(net.state_dict())
            elif self.args.dataset == 'EyePACS':
                epoch_val_auc = roc_auc_score(np.array(targets), np.array(prob), multi_class='ovo', labels=[0, 1, 2, 3, 4])
                val_aucs.append(epoch_val_auc)
                if epoch_val_auc >= best_val_auc:
                    best_val_auc = epoch_val_auc
                    best_model = copy.deepcopy(net.state_dict())
            else:
                if epoch_val_acc >= best_val_acc:
                    best_val_acc = epoch_val_acc
                    best_model = copy.deepcopy(net.state_dict())
            
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
                else:
                    for i in range(len(out)):
                        prob.append(F.softmax(out[i].double(), dim=0).cpu().tolist())
                targets += target.cpu().tolist()

                _, pred_label = torch.max(out.data, 1)
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

        test_acc = correct / float(total)

        if self.args.dataset not in {'isic2020', 'EyePACS'}:
            return test_acc
        else:
            if self.args.dataset == 'isic2020':
                test_auc = roc_auc_score(np.array(targets), np.array(prob))
            else:
                test_auc = roc_auc_score(np.array(targets), np.array(prob), multi_class='ovo', labels=[0, 1, 2, 3, 4])
            return test_auc
