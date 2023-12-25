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
from .feddistill import FedDistill
from dataset_utils import TensorDataset
import torch.nn.functional as F
from utils import get_dataloader, DatasetSplit, get_network, DiffAugment, match_loss, augment, ParamDiffAug

class Ours(FedDistill):
    def __init__(self, args, appr_args, logger):
        super(Ours, self).__init__(args, appr_args, logger)

    # function that processing the special arguments of current method
    @staticmethod
    def extra_parser(extra_args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--init', type=str, default='real',
                            help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
        parser.add_argument('--aug_iter', type=int, default=100,
                            help='number of iterations to apply augmentation')
        parser.add_argument('--aug_step', type=int, default=5,
                            help='number of augmentation steps')
        parser.add_argument('--lr_aug', type=float, default=1.0,
                            help='learning rate for optimizing aug data')
        parser.add_argument('--mom_aug', type=float, default=0.9,
                            help='momentum for optimizing aug data')
        parser.add_argument('--wd_aug', type=float, default=1e-5,
                            help='weight decay for optimizing aug data')
        parser.add_argument('--lr_center', type=float, default=1.0,
                            help='learning rate for optimizing class center')
        parser.add_argument('--mom_center', type=float, default=0.9,
                            help='momentum for optimizing class center')
        parser.add_argument('--wd_center', type=float, default=1e-5,
                            help='weight decay for optimizing class center')
        parser.add_argument('--ipc', type=int, default=1,
                            help='number of images distilled per class')
        parser.add_argument('--lr_img', type=float, default=1.0, 
                            help='learning rate for updating synthetic images')
        parser.add_argument('--iter', type=int, default=300,
                            help='distilling iterations')
        parser.add_argument('--batch_real', type=int, default=64, 
                            help='batch size for real data')
        parser.add_argument('--dis_metric', type=str, default='ours', 
                            help='distance metric, especially for DC or DSA method')
        parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', 
                            help='differentiable Siamese augmentation strategy')
        parser.add_argument('--aug_bs', type=int, default=64,
                            help='batch size for augmentation data')

        return parser.parse_args(extra_args)
    
    def run(self):

        if self.args.partition == 'noniid':
            self.appr_args.init = 'noise'
        self.appr_args.dsa_param = ParamDiffAug()
        self.appr_args.dsa = False if self.appr_args.dsa_strategy in ['none', 'None'] else True

        self.logger.info('Partitioning data...')
        
        train_ds, val_ds, test_ds, num_per_class = get_dataloader(self.args, request='dataset')
        self.party2dataidx = self.partition(train_ds, val_ds)

        self.logger.info('Initialize nets...')

        # According to the paper, MNIST and CIFAR-10 datasets are too small for ResNet to generate condensed images, so use ConvNet as the base model
        if self.args.dataset in {'mnist', 'cifar10'}:
            model = self.args.model
            self.args.model = 'ConvNet'
            local_nets = [get_network(self.args) for _ in range(self.args.C)]
            self.args.model = model
        else:
            local_nets = [get_network(self.args) for _ in range(self.args.C)]
        global_net = get_network(self.args)

        self.logger.info('Training begins...')

        syn_data = {
            'images': [],
            'label': []
        }

        for round in range(self.args.n_comm_round):
            
            self.logger.info('Communication Round: %d' % round)
            party_list_this_round = random.sample([_ for _ in range(self.args.C)], 
                                                int(self.args.C * self.args.sample_fraction))
            party_list_this_round.sort()

            for client_idx in party_list_this_round:

                self.logger.info('Client %d' % client_idx)
                
                train_ds_c = DatasetSplit(train_ds, self.party2dataidx['train'][client_idx])
                val_ds_c = DatasetSplit(val_ds, self.party2dataidx['val'][client_idx])
                train_dl = DataLoader(train_ds_c, num_workers=8, prefetch_factor=2*self.args.train_bs,
                                    batch_size=self.args.train_bs, shuffle=True, drop_last=False, pin_memory=True)
                val_dl = DataLoader(val_ds_c, num_workers=8, prefetch_factor=2*self.args.test_bs,
                                    batch_size=self.args.test_bs, shuffle=False, pin_memory=True)
                self.logger.info('Train batches: %d' % len(train_dl))
                self.logger.info('Val batches: %d' % len(val_dl))

                if self.args.device != 'cpu':
                    local_nets[client_idx] = nn.DataParallel(local_nets[client_idx])
                    local_nets[client_idx].to(self.args.device)
                
                self.logger.info('Organize the real dataset')
                labels_all = [train_ds.targets[i] for i in self.party2dataidx['train'][client_idx]]
                indices_class = [[] for _ in range(self.args.n_classes)]
                for _, lab in enumerate(labels_all):
                    indices_class[lab].append(_)

                for _ in range(self.args.n_classes):
                    self.logger.info('class c = %d: %d real images' % (_, len(indices_class[_])))

                self.logger.info('Initialize synthetic data')
                image_syn = torch.randn(size=(self.args.n_classes * self.appr_args.ipc, self.args.channel, 
                                        self.args.im_size[0], self.args.im_size[1]), dtype=torch.float, requires_grad=True)
                label_syn = torch.tensor(np.array([np.ones(self.appr_args.ipc)*i for i in range(self.args.n_classes)]),
                                        dtype=torch.long, requires_grad=False).view(-1)
                if self.appr_args.init == 'real':
                    for c in range(self.args.n_classes):
                        image_syn.data[c*self.appr_args.ipc: (c+1)*self.appr_args.ipc] = self.get_images(c, self.appr_args.ipc, indices_class, train_ds_c).detach().data
                elif self.appr_args.init == 'noise':
                    pass

                best_img_syn, best_lab_syn = self.CADA(local_nets[client_idx], indices_class, image_syn, label_syn, train_ds, train_dl, val_dl)

                self.save(best_img_syn, best_lab_syn, client_idx)
                self.visualize(best_img_syn, client_idx)

                for i in range(len(best_img_syn)):
                    syn_data['images'].append(best_img_syn[i].detach().cpu())
                    syn_data['label'].append(best_lab_syn[i].detach().cpu())

            if self.args.device != 'cpu':
                global_net = nn.DataParallel(global_net)
                global_net.to(self.args.device)

            global_w = self.global_train(copy.deepcopy(global_net), syn_data)
            global_net.load_state_dict(global_w)
            if (round + 1) % self.args.save_interval == 0 or round == 0:
                torch.save(global_w,
                    os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'global_{}_round{}_'.format(self.args.model, round)+self.args.log_file_name+'.pth'))
            
            test_dl = DataLoader(dataset=test_ds, batch_size=self.args.test_bs, shuffle=False,
                                num_workers=8, pin_memory=True, prefetch_factor=16*self.args.test_bs)
            
            if self.args.dataset not in {'isic2020', 'EyePACS'}:
                test_acc = self.eval(global_net, test_dl)
                self.logger.info('>>> Global Model Test Accuracy: %f' % test_acc)
            else:
                test_auc = self.eval(global_net, test_dl)
                self.logger.info('>>> Global Model Test AUC: %f' % test_auc)

    def compute_loss(self, x, y, mode='mse'):

        dis = torch.tensor(0.0).to(self.args.device)
        
        if mode == 'mse':
            x_vec = []
            y_vec = []
            for i in range(len(x)):
                x_vec.append(x[i].reshape((-1)))
                y_vec.append(y[i].reshape((-1)))
            x_vec = torch.cat(x_vec, dim=0)
            y_vec = torch.cat(y_vec, dim=0)
            dis = torch.sum((x_vec - y_vec) ** 2)
        elif mode == 'ac':
            x_vec = []
            y_vec = []
            for i in range(len(x)):
                x_vec.append(x[i].reshape((-1)))
                y_vec.append(y[i].reshape((-1)))
            x_vec = torch.cat(x_vec, dim=0)
            y_vec = torch.cat(y_vec, dim=0)
            dis = torch.sum(x_vec * y_vec) / (torch.norm(x_vec) * torch.norm(y_vec) + 0.000001)
            dis = torch.arccos(dis) / torch.pi

        return dis

    def CADA(self, net, indices_class, image_syn, label_syn, train_ds, train_dl, val_dl):

        optimizer_img = optim.SGD([image_syn, ], lr=self.appr_args.lr_img, momentum=0.5)
        optimizer_net = optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                  weight_decay=self.args.weight_decay)
        optimizer_img.zero_grad()
        optimizer_net.zero_grad()
        criterion = nn.CrossEntropyLoss()

        best_metric = 0
        best_img_syn, best_lab_syn = None, None

        class_centers = torch.randn(size=(self.args.n_classes, self.args.channel, self.args.im_size[0], self.args.im_size[1]), 
                                    dtype=torch.float, requires_grad=True, device=self.args.device)
        aug_data = {
            'images': [],
            'label': []
        }
        aug_dst = None
        aug_indices_class = [[] for _ in range(self.args.n_classes)]

        for it in range(self.appr_args.iter):

            net.train()
            net_parameters = list(net.parameters())

            ''' freeze the running mu and sigma for BatchNorm layers '''

            BN_flag = False
            BNSizePC = 16
            for module in net.modules():
                if 'BatchNorm' in module._get_name():
                    BN_flag = True
            if BN_flag:
                img_real = torch.cat([self.get_images(c, BNSizePC, indices_class, train_ds) for c in range(self.args.n_classes)], dim=0)
                net.train()
                output_real = net(img_real)
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():
                        module.eval()

            ''' CADA Domain Generalization '''
            if it % self.appr_args.aug_iter == 0:

                if len(aug_data['images']) > 0:
                    aug_dst_tmp = TensorDataset(torch.stack(aug_data['images'], dim=0), torch.stack(aug_data['label'], dim=0))
                    aug_dl = DataLoader(aug_dst_tmp, num_workers=8, prefetch_factor=2*self.appr_args.aug_bs,
                                        batch_size=self.appr_args.aug_bs, shuffle=True, drop_last=False, pin_memory=True)
                    for batch_idx, (x, target) in enumerate(aug_dl):
                        x_aug = copy.deepcopy(x)
                        x = x.to(self.args.device, non_blocking=True)
                        x_aug = x_aug.to(self.args.device, non_blocking=True)
                        optimizer_aug = optim.SGD([x_aug, ], lr=self.appr_args.lr_aug, momentum=self.appr_args.mom_aug,
                                                weight_decay=self.appr_args.wd_aug)
                        for t in range(self.appr_args.aug_step):
                            loss_exp = torch.tensor(0.0).to(self.args.device)
                            loss_exp += self.compute_loss(x_aug, x, 'mse')
                            loss_ac = torch.tensor(0.0).to(self.args.device)
                            for i in range(len(x)):
                                loss_ac += self.compute_loss(x_aug[i], class_centers[target[i]], 'ac')
                            loss_ac = loss_ac / len(x)
                            loss_exp -= loss_ac

                            optimizer_aug.zero_grad()
                            loss_exp.backward()
                            optimizer_aug.step()
                    for i in range(len(x_aug)):
                        aug_data['images'].append(x_aug[i].detach().cpu())
                        aug_data['label'].append(target[i].detach().cpu())
                        aug_indices_class[target[i]].append(len(aug_data['label'])-1)

                for batch_idx, (x, target) in enumerate(train_dl):
                    x_aug = copy.deepcopy(x)
                    x = x.to(self.args.device, non_blocking=True)
                    x_aug = x_aug.to(self.args.device, non_blocking=True)
                    optimizer_aug = optim.SGD([x_aug, ], lr=self.appr_args.lr_aug, momentum=self.appr_args.mom_aug,
                                              weight_decay=self.appr_args.wd_aug)
                    for t in range(self.appr_args.aug_step):
                        loss_exp = torch.tensor(0.0).to(self.args.device)
                        loss_exp += self.compute_loss(x_aug, x, 'mse')
                        loss_ac = torch.tensor(0.0).to(self.args.device)
                        for i in range(len(x)):
                            loss_ac += self.compute_loss(x_aug[i], class_centers[target[i]], 'ac')
                        loss_ac = loss_ac / len(x)
                        loss_exp -= loss_ac

                        optimizer_aug.zero_grad()
                        loss_exp.backward()
                        optimizer_aug.step()
                for i in range(len(x_aug)):
                    aug_data['images'].append(x_aug[i].detach().cpu())
                    aug_data['label'].append(target[i].detach().cpu())
                    aug_indices_class[target[i]].append(len(aug_data['label'])-1)
                aug_dst = TensorDataset(torch.stack(aug_data['images'], dim=0), torch.stack(aug_data['label'], dim=0))

            ''' Update Synthetic Data '''
            loss = torch.tensor(0.0).to(self.args.device)
            for c in range(self.args.n_classes):
                if len(indices_class[c]) == 0:
                    continue

                if len(aug_indices_class[c]) > 0:
                    img_real = self.get_images(c, self.appr_args.batch_real, aug_indices_class, aug_dst)
                else:
                    img_real = self.get_images(c, self.appr_args.batch_real, indices_class, train_ds)
                lab_real = torch.ones((img_real.shape[0], ), device=self.args.device, dtype=torch.long) * c
                img_syn = image_syn[c*self.appr_args.ipc: (c+1)*self.appr_args.ipc].reshape((self.appr_args.ipc, self.args.channel, self.args.im_size[0], self.args.im_size[1]))
                lab_syn = torch.ones((self.appr_args.ipc, ), device=self.args.device, dtype=torch.long) * c

                if self.appr_args.dsa:
                    seed = 42
                    img_real = DiffAugment(img_real, self.appr_args.dsa_strategy, seed=seed, param=self.appr_args.dsa_param)
                    img_syn = DiffAugment(img_syn, self.appr_args.dsa_strategy, seed=seed, param=self.appr_args.dsa_param)

                output_real = net(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))

                output_syn = net(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                loss += match_loss(gw_syn, gw_real, self.args, self.appr_args)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

            ''' Update Network and Class Center'''
            image_syn_train, label_syn_train = image_syn.detach(), label_syn.detach()
            dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
            trainloader = DataLoader(dst_syn_train, num_workers=8, prefetch_factor=2*self.args.train_bs,
                                    batch_size=self.args.train_bs, shuffle=True, drop_last=False, pin_memory=True)
            
            if self.args.dataset in ['isic2020', 'EyePACS']:
                epoch_train_loss, epoch_train_auc = self.epoch('train', trainloader, net, optimizer_net, criterion, class_centers, aug=True if self.appr_args.dsa else False)
                epoch_val_loss, epoch_val_auc = self.epoch('val', val_dl, net, optimizer_net, criterion, class_centers, aug=False)
                self.logger.info('Iter: %d Synthetic Train loss: %f Train AUC: %f Val loss: %f Val AUC: %f' % (it, epoch_train_loss, epoch_train_auc, epoch_val_loss, epoch_val_auc))
                if epoch_val_auc > best_metric:
                    best_metric = epoch_val_auc
                    best_img_syn = image_syn
                    best_lab_syn = label_syn
            else:
                epoch_train_loss, epoch_train_acc = self.epoch('aug_train', trainloader, net, optimizer_net, criterion, class_centers, aug=True if self.appr_args.dsa else False)
                epoch_val_loss, epoch_val_acc = self.epoch('val', val_dl, net, optimizer_net, criterion, class_centers, aug=False)
                self.logger.info('Iter: %d Synthetic Train loss: %f Train Acc: %f Val loss: %f Val Acc: %f' % (it, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc))
                if epoch_val_acc > best_metric:
                    best_metric = epoch_val_acc
                    best_img_syn = image_syn
                    best_lab_syn = label_syn

        return best_img_syn, best_lab_syn
    
    def epoch(self, mode, dataloader, net, optimizer, criterion, class_centers, aug):

        epoch_loss_collector = []
        total, correct = 0, 0
        net.to(self.args.device)

        if mode == 'train' or mode == 'aug_train':
            net.train()
            optimizer_center = optim.SGD([class_centers, ], lr=self.appr_args.lr_center, momentum=self.appr_args.mom_center,
                                         weight_decay=self.appr_args.wd_center)
        else:
            net.eval()
        
        y_true, y_prob = [], []
        
        for batch_idx, (x, target) in enumerate(dataloader):
            
            x = x.to(self.args.device, non_blocking=True)
            if aug:
                if self.appr_args.dsa:
                    x = DiffAugment(x, self.appr_args.dsa_strategy, param=self.appr_args.dsa_param)
                else:
                    x = augment(x, self.appr_args.dc_aug_param, device=self.args.device)
            target = target.long().to(self.args.device, non_blocking=True)
            out = net(x)

            loss = criterion(out, target)
            if mode == 'aug_train':
                for i in range(len(x)):
                    loss += self.compute_loss(x[i], class_centers[target[i]], mode='ac')
            total += x.data.size()[0]
            _, pred_label = torch.max(out.data, 1)
            if self.args.dataset == 'isic2020':
                for i in range(len(out)):
                    y_prob.append(F.softmax(out[i], dim=0).cpu().tolist()[1])
            elif self.args.dataset == 'EyePACS':
                for i in range(len(out)):
                    y_prob.append(F.softmax(out[i], dim=0).cpu().tolist())
            y_true += target.cpu().tolist()
            correct += (pred_label == target.data).sum().item()
            
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            elif mode == 'aug_train':
                optimizer.zero_grad()
                optimizer_center.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_center.step()
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_acc = correct / float(total)
        if self.args.dataset == 'isic2020':
            epoch_auc = roc_auc_score(np.array(y_true), np.array(y_prob))
        elif self.args.dataset == 'EyePACS':
            epoch_auc = roc_auc_score(np.array(y_true), np.array(y_prob), multi_class='ovo', labels=[0, 1, 2, 3, 4])
            
        if self.args.dataset in {'isic2020', 'EyePACS'}:
            return epoch_loss, epoch_auc
        else:
            return epoch_loss, epoch_acc