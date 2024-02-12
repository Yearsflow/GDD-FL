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
from utils.dataset_utils import TensorDataset
from torchvision.utils import save_image
import torch.nn.functional as F
from utils.common_utils import get_dataloader, DatasetSplit, get_network, get_loops, ParamDiffAug, DiffAugment, match_loss, augment

class FedDistill(FedAvg):
    def __init__(self, args, appr_args, logger):
        super(FedDistill, self).__init__(args, appr_args, logger)

    # function that processing the special arguments of current method
    @staticmethod
    def extra_parser(extra_args):
        parser = argparse.ArgumentParser()

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
        parser.add_argument('--init', type=str, default='noise', 
                            help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

        return parser.parse_args(extra_args)
    
    def run(self):
        
        self.appr_args.dsa_param = ParamDiffAug()
        self.appr_args.dsa = False if self.appr_args.dsa_strategy in ['none', 'None'] else True
        if self.args.approach == 'feddc':
            self.appr_args.dsa = False
        elif self.args.approach == 'feddsa':
            self.appr_args.dsa = True
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
    
            for c_id in party_list_this_round:
                self.logger.info('Client %d' % c_id)

                train_ds_c = DatasetSplit(train_ds, self.party2dataidx['train'][c_id])
                val_ds_c = DatasetSplit(val_ds, self.party2dataidx['val'][c_id])

                train_dl = DataLoader(train_ds_c, num_workers=8, prefetch_factor=2*self.args.train_bs,
                                    batch_size=self.args.train_bs, shuffle=True, drop_last=False, pin_memory=True)
                val_dl = DataLoader(val_ds_c, num_workers=8, prefetch_factor=2*self.args.test_bs,
                                    batch_size=self.args.test_bs, shuffle=False, pin_memory=True)
                self.logger.info('Train batches: %d' % len(train_dl))
                self.logger.info('Val batches: %d' % len(val_dl))

                if self.args.device != 'cpu':
                    local_nets[c_id] = nn.DataParallel(local_nets[c_id])
                    local_nets[c_id].to(self.args.device)

                self.logger.info('Organize the real dataset')
                labels_all = [train_ds.targets[i] for i in self.party2dataidx['train'][c_id]]
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
                
                if self.args.approach in {'feddc', 'feddsa'}:
                    best_img_syn, best_lab_syn = self.DC(local_nets[c_id], indices_class, image_syn, label_syn, train_ds_c, val_ds_c)
                elif self.args.approach == 'feddm':
                    best_img_syn, best_lab_syn = self.DM(local_nets[c_id], indices_class, image_syn, label_syn, train_ds_c)

                self.save(best_img_syn, best_lab_syn, c_id)
                self.visualize(best_img_syn, c_id)

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

        losses, accs, w = self.train(net, train_dl, val_dl)
        return w

    def get_images(self, c, n, indices_class, train_ds):

        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        images = []
        for idx in idx_shuffle:
            images.append(torch.unsqueeze(train_ds[idx][0], dim=0))
        return torch.cat(images, dim=0).to(self.args.device, non_blocking=True)

    def DC(self, net, indices_class, image_syn, label_syn, train_ds, val_ds):

        optimizer_img = torch.optim.SGD([image_syn, ], lr=self.appr_args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss()

        best_loss = 1e8
        best_img_syn, best_lab_syn = None, None

        for it in range(self.appr_args.iter):
            
            net.train()
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
                    img_real = torch.cat([self.get_images(c, BNSizePC, indices_class, train_ds) for c in range(self.args.n_classes)], dim=0)
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

                    img_real = self.get_images(c, self.appr_args.batch_real, indices_class, train_ds)
                    lab_real = torch.ones((img_real.shape[0],), device=self.args.device, dtype=torch.long) * c
                    img_syn = image_syn[c*self.appr_args.ipc: (c+1)*self.appr_args.ipc].reshape((self.appr_args.ipc, self.args.channel, self.args.im_size[0], self.args.im_size[1]))
                    lab_syn = torch.ones((self.appr_args.ipc,), device=self.args.device, dtype=torch.long) * c

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
    
    def DM(self, net, indices_class, image_syn, label_syn, train_ds):

        optimizer_img = torch.optim.SGD([image_syn, ], lr=self.appr_args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()

        best_loss = 1e8
        best_img_syn, best_lab_syn = None, None

        for it in range(self.appr_args.iter):

            ''' Train synthetic data '''
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False
            embed = net.module.embed

            loss_avg = 0

            ''' update synthetic data '''
            loss = torch.tensor(0.0).to(self.args.device)
            for c in range(self.args.n_classes):

                if len(indices_class[c]) == 0:
                    continue

                img_real = self.get_images(c, self.appr_args.batch_real, indices_class, train_ds)
                img_syn = image_syn[c*self.appr_args.ipc: (c+1)*self.appr_args.ipc].reshape((self.appr_args.ipc, self.args.channel, self.args.im_size[0], self.args.im_size[1]))

                if self.appr_args.dsa:
                    seed = 42
                    img_real = DiffAugment(img_real, self.appr_args.dsa_strategy, seed=seed, param=self.appr_args.dsa_param)
                    img_syn = DiffAugment(img_syn, self.appr_args.dsa_strategy, seed=seed, param=self.appr_args.dsa_param)

                output_real = embed(img_real).detach()
                img_syn = img_syn.to(self.args.device, non_blocking=True)
                output_syn = embed(img_syn)

                loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()
            loss_avg /= self.args.n_classes

            if loss_avg <= best_loss:
                best_loss = loss_avg
                best_img_syn = image_syn
                best_lab_syn = label_syn

            if it % 10 == 0:
                self.logger.info('iter: %05d loss: %.4f' % (it, loss_avg))

        return best_img_syn, best_lab_syn
 
    def save(self, best_img_syn, best_lab_syn, c_id):
        
        data_save = []
        data_save.append([copy.deepcopy(best_img_syn.detach().cpu()), copy.deepcopy(best_lab_syn.detach().cpu())])
        torch.save(data_save,
            os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'res_client{}_'.format(c_id)+self.args.log_file_name+'.pth'))

    def visualize(self, best_img_syn, c_id):

        save_name = os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'vis_client{}_'.format(c_id)+self.args.log_file_name+'.png')
        image_syn_vis = copy.deepcopy(best_img_syn.detach().cpu())
        for ch in range(self.args.channel):
            image_syn_vis[:, ch] = image_syn_vis[:, ch] * self.args.std[ch] + self.args.mean[ch]
        image_syn_vis[image_syn_vis < 0] = 0.0
        image_syn_vis[image_syn_vis > 1] = 1.0
        save_image(image_syn_vis, save_name, nrow=self.appr_args.ipc)

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
            out = net(x)
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