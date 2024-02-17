import torch
import os
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, ConcatDataset
import copy
import random
from .feddistill import FedDistill
from utils.dataset_utils import TensorDataset
import torch.nn.functional as F
from utils.common_utils import get_dataloader, DatasetSplit, get_network, DiffAugment, match_loss, augment, ParamDiffAug, get_loops
from networks import AugNet, ConvNet, ResNet
import albumentations as A
from utils.fedl2d_contrastive_loss import SupConLoss
from torchvision import transforms
from utils.fedl2d_utils import loglikeli, club, conditional_mmd_rbf
import math

class FedL2D(FedDistill):
    def __init__(self, args, appr_args, logger):
        super(FedL2D, self).__init__(args, appr_args, logger)

    # function that processing the special arguments of current method
    @staticmethod
    def extra_parser(extra_args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--init', type=str, default='noise',
                            help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
        parser.add_argument('--ipc', type=int, default=10,
                            help='number of images distilled per class')
        parser.add_argument('--lr_img', type=float, default=1.0, 
                            help='learning rate for updating synthetic images')
        parser.add_argument('--iter', type=int, default=100,
                            help='distilling iterations')
        parser.add_argument('--batch_real', type=int, default=64, 
                            help='batch size for real data')
        parser.add_argument('--dis_metric', type=str, default='ours', 
                            help='distance metric, especially for DC or DSA method')
        parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', 
                            help='differentiable Siamese augmentation strategy')
        parser.add_argument('--aug_epochs', type=int, default=30,
                            help='number of augmentation epochs')
        parser.add_argument('--aug_lr', type=float, default=0.001,
                            help='learning rate for augmentation')
        parser.add_argument("--alpha1", default=1, type=float,
                            help='hyper-parameter for balancing loss')
        parser.add_argument("--alpha2", default=1, type=float,
                            help='hyper-parameter for balancing loss')
        parser.add_argument('--con_lr', type=float, default=10, 
                            help='learning rate for convertor')
        parser.add_argument("--beta", default=0.1, type=float,
                            help='balancing weight')
        parser.add_argument('--gamma', type=float, default=0.9,
                            help='balancing distribution loss and grad loss')

        return parser.parse_args(extra_args)

    def L2D(self, train_dl, val_dl):

        convertor = AugNet(self.args.im_size[0])
        if self.args.device != 'cpu':
            convertor = nn.DataParallel(convertor)
            convertor.to(self.args.device)
        optimizer_convertor = optim.SGD(convertor.parameters(), lr=self.appr_args.con_lr)
        extractor = get_network(self.args)
        if self.args.device != 'cpu':
            extractor = nn.DataParallel(extractor)
            extractor.to(self.args.device)
        optimizer_aug = optim.SGD(extractor.parameters(), lr=self.appr_args.aug_lr, nesterov=True, 
                                  momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer_aug, step_size=int(self.appr_args.aug_epochs * 0.8))
        transform = transforms.Normalize(mean=self.args.mean, std=self.args.std)
        con_loss = SupConLoss()

        self.logger.info('Learning to diversify')

        for ep in range(self.appr_args.aug_epochs):
            
            criterion = nn.CrossEntropyLoss()
            extractor.train()
            epoch_loss_collector = []
            total, correct_aug, correct = 0, 0, 0
            prob, prob_aug, targets = [], [], []
            aug_data = []

            for batch_idx, (x, target) in enumerate(train_dl):
                x = x.to(self.args.device, non_blocking=True)
                target = target.long().to(self.args.device, non_blocking=True)

                # Aug
                optimizer_aug.zero_grad()
                inputs_max = transform(torch.sigmoid(convertor(x)))
                inputs_max = inputs_max * 0.6 + x * 0.4
                x_aug = torch.cat([inputs_max, x])
                labels = torch.cat([target, target])

                # forward
                logits, tuple = extractor(x_aug)
                if self.args.dataset == 'isic2020':
                    for i in range(target.size(0)):
                        prob_aug.append(F.softmax(logits[i], dim=0).cpu().tolist()[1])
                    for i in range(target.size(0), len(logits)):
                        prob.append(F.softmax(logits[i], dim=0).cpu().tolist()[1])
                elif self.args.dataset == 'EyePACS':
                    for i in range(target.size(0)):
                        prob_aug.append(F.softmax(logits[i], dim=0).cpu().tolist())
                    for i in range(target.size(0), len(logits)):
                        prob.append(F.softmax(logits[i], dim=0).cpu().tolist())
                targets += target.cpu().tolist()

                # Maximize MI between z and z_hat
                emb_src = F.normalize(tuple['Embedding'][:target.size(0)]).unsqueeze(1)
                emb_aug = F.normalize(tuple['Embedding'][target.size(0):]).unsqueeze(1)
                con = con_loss(torch.cat([emb_src, emb_aug], dim=1), target)

                # Likelihood
                mu = tuple['mu'][target.size(0):]
                logvar = tuple['logvar'][target.size(0):]
                y_samples = tuple['Embedding'][:target.size(0)]
                likeli = -loglikeli(mu, logvar, y_samples)

                # Total loss & backward
                class_loss = criterion(logits, labels)
                loss = class_loss + self.appr_args.alpha2 * likeli + self.appr_args.alpha1 * con
                loss.backward()
                optimizer_aug.step()
                epoch_loss_collector.append(class_loss.item())
                _, cls_pred = logits.max(dim=1)

                inputs_max = transform(torch.sigmoid(convertor(x, estimation=True)))
                inputs_max = inputs_max * 0.6 + x * 0.4
                x_aug = torch.cat([inputs_max, x])
                aug_data.append(inputs_max.detach())

                # forward with the adapted parameters
                outputs, tuples = extractor(x_aug)

                # Upper bound MI
                mu = tuples['mu'][target.size(0):]
                logvar = tuples['logvar'][target.size(0):]
                y_samples = tuples['Embedding'][:target.size(0)]
                div = club(mu, logvar, y_samples)

                # Semantic consistency
                e = tuples['Embedding']
                e1 = e[:target.size(0)]
                e2 = e[target.size(0):]
                dist = conditional_mmd_rbf(e1, e2, target, num_class=self.args.n_classes)

                # Total loss & backward
                optimizer_convertor.zero_grad()
                (dist + self.appr_args.beta * div).backward()
                optimizer_convertor.step()

                total += target.shape[0]
                correct_aug += torch.sum(cls_pred[:target.size(0)] == target.data).item()
                correct += torch.sum(cls_pred[target.size(0):] == target.data).item()

            scheduler.step()

            # Compute metric
            epoch_train_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_train_acc = correct / float(total)
            epoch_train_aug_acc = correct_aug / float(total)
            if self.args.dataset == 'isic2020':
                epoch_train_auc = roc_auc_score(np.array(targets), np.array(prob))
                epoch_train_aug_auc = roc_auc_score(np.array(targets), np.array(prob_aug))
            elif self.args.dataset == 'EyePACS':
                epoch_train_auc = roc_auc_score(np.array(targets), np.array(prob), multi_class='ovo', labels=[0, 1, 2, 3, 4])
                epoch_train_aug_auc = roc_auc_score(np.array(targets), np.array(prob_aug), multi_class='ovo', labels=[0, 1, 2, 3, 4])

            del loss, class_loss, logits

            # Evaluation
            best_metric = 0
            best_data = None
            if self.args.dataset in ['mnist', 'cifar10']:
                val_acc = self.eval(extractor, val_dl)
                if val_acc > best_metric:
                    best_metric = val_acc
                    best_data = copy.deepcopy(aug_data)
                self.logger.info('Epoch: %d loss: %f Train Acc w Aug: %f Train Acc w/o Aug: %f Val Acc: %f' %
                                 (ep, epoch_train_loss, epoch_train_aug_acc, epoch_train_acc, val_acc))
            else:
                val_auc = self.eval(extractor, val_dl)
                if val_auc > best_metric:
                    best_metric = val_auc
                    best_data = copy.deepcopy(aug_data)
                self.logger.info('Epoch: %d loss: %f Train AUC w Aug: %f Train AUC w/o Aug: %f Val AUC: %f' %
                                 (ep, epoch_train_loss, epoch_train_aug_auc, epoch_train_auc, val_auc))
                
        return best_data, targets
    
    def epoch(self, mode, dataloader, net, optimizer, criterion, aug):

        epoch_loss_collector = []
        total, correct = 0, 0
        net = net.to(self.args.device)

        if mode == 'train':
            net.train()
        else:
            net.eval()
        
        y_true, y_pred, y_prob = [], [], []
        
        for batch_idx, (x, target) in enumerate(dataloader):
            
            x = x.to(self.args.device, non_blocking=True)
            if aug:
                if self.appr_args.dsa:
                    x = DiffAugment(x, self.appr_args.dsa_strategy, param=self.appr_args.dsa_param)
                else:
                    x = augment(x, self.appr_args.dc_aug_param, device=self.args.device)
            target = target.long().to(self.args.device, non_blocking=True)
            out = net(x)[0]

            loss = criterion(out, target)
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

    def eval(self, net, test_dl):

        total, correct = 0, 0
        prob, targets = [], []     

        net.eval()

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_dl):
                x = x.to(self.args.device, non_blocking=True)
                target = target.long().to(self.args.device, non_blocking=True)
                with torch.cuda.amp.autocast_mode.autocast():
                    out = net(x)[0]

                if self.args.dataset == 'isic2020':
                    for i in range(len(out)):
                        prob.append(F.softmax(out[i], dim=0).cpu().tolist()[1])
                elif self.args.dataset == 'EyePACS':
                    for i in range(len(out)):
                        prob.append(F.softmax(out[i].double(), dim=0).cpu().tolist())
                targets += target.cpu().tolist()

                _, pred_label = torch.max(out.data, 1)
                pred_label = pred_label.cpu()
                total += x.data.size()[0]
                correct += (pred_label == target.data.cpu()).sum().item()

        test_acc = correct / float(total)
        if self.args.dataset == 'isic2020':
            test_auc = roc_auc_score(np.array(targets), np.array(prob))
        elif self.args.dataset == 'EyePACS':
            test_auc = roc_auc_score(np.array(targets), np.array(prob), multi_class='ovo', labels=[0, 1, 2, 3, 4])

        if self.args.dataset not in {'isic2020', 'EyePACS'}:
            return test_acc
        else:
            return test_auc
        
    def global_train(self, net, data):

        n = len(data['images']) # ipc * C * n_classes
        train_num = self.appr_args.ipc * (self.args.C - 1) * self.args.n_classes
        val_num = n - train_num
        self.logger.info('Total %d samples for global training' % train_num)
        self.logger.info('Total %d samples for global validation' % val_num)

        train_images = data['images'][:train_num]
        train_labels = data['label'][:train_num]
        val_images = data['images'][train_num:]
        val_labels = data['label'][train_num:]

        dst_train = TensorDataset(torch.stack(train_images, dim=0), torch.stack(train_labels, dim=0))
        dst_val = TensorDataset(torch.stack(val_images, dim=0), torch.stack(val_labels, dim=0))

        train_dl = DataLoader(dst_train, num_workers=8, prefetch_factor=2*self.args.train_bs,
                            batch_size=self.args.train_bs, shuffle=True, drop_last=False, pin_memory=True)
        val_dl = DataLoader(dst_val, num_workers=8, prefetch_factor=2*self.args.test_bs,
                            batch_size=self.args.test_bs, shuffle=False, pin_memory=True)

        optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-5)
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
                    out = net(x)[0]
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
                        out = net(x)[0]
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

        return best_model
        
    def condense(self, net, E, indices_class, image_syn, label_syn, aug_ds, val_ds):

        optimizer_img = torch.optim.SGD([image_syn, ], lr=self.appr_args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss()

        best_loss = 1e8
        best_img_syn, best_lab_syn = None, None

        for it in range(self.appr_args.iter):
            
            net.train()
            E.train()
            for param in list(E.parameters()):
                param.requires_grad = False
            embed = E.module.embed
            net_parameters = list(net.parameters())
            optimizer_net = optim.SGD(net.parameters(), lr=self.args.lr)
            optimizer_net.zero_grad()
            self.appr_args.dc_aug_param = None

            loss_avg = 0

            for ol in range(self.appr_args.outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''

                BN_flag = False
                BNSizePC = 16
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([self.get_images(c, BNSizePC, indices_class, aug_ds) for c in range(self.args.n_classes)], dim=0)
                    net.train()
                    output_real = net(img_real)
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():
                            module.eval()

                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(self.args.device)
                for c in range(self.args.n_classes):

                    if len(indices_class[c]) == 0:
                        continue

                    img_real = self.get_images(c, self.appr_args.batch_real, indices_class, aug_ds)
                    lab_real = torch.ones((img_real.shape[0],), device=self.args.device, dtype=torch.long) * c
                    img_syn = image_syn[c*self.appr_args.ipc: (c+1)*self.appr_args.ipc].reshape((self.appr_args.ipc, self.args.channel, self.args.im_size[0], self.args.im_size[1]))
                    lab_syn = torch.ones((self.appr_args.ipc,), device=self.args.device, dtype=torch.long) * c

                    if self.appr_args.dsa:
                        seed = 42
                        img_real = DiffAugment(img_real, self.appr_args.dsa_strategy, seed=seed, param=self.appr_args.dsa_param)
                        img_syn = DiffAugment(img_syn, self.appr_args.dsa_strategy, seed=seed, param=self.appr_args.dsa_param)

                    feature_real = embed(img_real).detach()
                    img_syn = img_syn.to(self.args.device, non_blocking=True)
                    feature_syn = embed(img_syn)

                    loss += torch.sum((torch.mean(feature_real, dim=0) - torch.mean(feature_syn, dim=0)) ** 2) * (1 - self.appr_args.gamma)

                    output_real = net(img_real)[0]
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)[0]
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss += match_loss(gw_syn, gw_real, self.args, self.appr_args) * self.appr_args.gamma
                
                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                if ol == self.appr_args.outer_loop - 1:
                    break

                ''' update network '''
                image_syn_train, label_syn_train = image_syn.detach(), label_syn.detach()
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = DataLoader(dst_syn_train, num_workers=8, prefetch_factor=2*self.args.train_bs,
                                        batch_size=self.args.train_bs, shuffle=True, drop_last=False, pin_memory=True)
                valloader = DataLoader(val_ds, num_workers=8, prefetch_factor=2*self.args.test_bs,
                                       batch_size=self.args.test_bs, shuffle=False, pin_memory=True)
                for il in range(self.appr_args.inner_loop):
                    if self.args.dataset in {'isic2020', 'EyePACS'}:
                        epoch_train_loss, epoch_train_auc = self.epoch('train', trainloader, net, optimizer_net, criterion, aug=True if self.appr_args.dsa else False)
                        epoch_val_loss, epoch_val_auc = self.epoch('val', valloader, net, optimizer_net, criterion, aug=False)
                        self.logger.info('Inner Loop: %d Train loss: %f Train AUC: %f Val loss: %f Val AUC: %f' % (il, epoch_train_loss, epoch_train_auc, epoch_val_loss, epoch_val_auc))
                    else:
                        epoch_train_loss, epoch_train_acc = self.epoch('train', trainloader, net, optimizer_net, criterion, aug=True if self.appr_args.dsa else False)
                        epoch_val_loss, epoch_val_acc = self.epoch('val', valloader, net, optimizer_net, criterion, aug=False)
                        self.logger.info('Inner Loop: %d Train loss: %f Train Acc: %f Val loss: %f Val Acc: %f' % (il, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc))
    
            loss_avg /= (self.args.n_classes * self.appr_args.outer_loop)

            if it % 10 == 0:
                self.logger.info('Iter: %04d loss: %.4f' % (it, loss_avg))

            if loss_avg <= best_loss:
                best_loss = loss_avg
                best_img_syn = image_syn
                best_lab_syn = label_syn

        return best_img_syn, best_lab_syn
    
    def run(self):

        self.appr_args.dsa_param = ParamDiffAug()
        self.appr_args.dsa = False
        self.appr_args.outer_loop, self.appr_args.inner_loop = get_loops(self.appr_args.ipc)

        self.logger.info('Partitioning data...')
        
        train_ds, val_ds, public_ds, test_ds, num_per_class = get_dataloader(self.args)
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

                best_data, best_label = self.L2D(train_dl, val_dl)
                extractor = get_network(self.args)
                if self.args.device != 'cpu':
                    extractor = nn.DataParallel(extractor)
                    extractor.to(self.args.device)
                aug_img = []
                for i in range(len(best_data)):
                    aug_img.append(best_data[i].detach().cpu())
                aug_dst = TensorDataset(torch.cat(aug_img, dim=0), torch.Tensor(best_label))
                mix_ds = ConcatDataset([train_ds_c, aug_dst])
                labels_all += best_label
                indices_class = [[] for _ in range(self.args.n_classes)]
                for _, lab in enumerate(labels_all):
                    indices_class[lab].append(_)

                best_img_syn, best_lab_syn = self.condense(local_nets[client_idx], extractor, indices_class, image_syn, label_syn, mix_ds, val_ds_c)

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
                                num_workers=8, pin_memory=True, prefetch_factor=2*self.args.test_bs)
            
            if self.args.dataset not in {'isic2020', 'EyePACS'}:
                test_acc = self.eval(global_net, test_dl)
                self.logger.info('>>> Global Model Test Accuracy: %f' % test_acc)
            else:
                test_auc = self.eval(global_net, test_dl)
                self.logger.info('>>> Global Model Test AUC: %f' % test_auc)