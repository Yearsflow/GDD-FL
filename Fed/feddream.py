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
from torchvision.utils import save_image
import torch.nn.functional as F
from utils import get_dataloader, DatasetSplit, get_network, get_loops
from networks import Generator, Discriminator
from torch.autograd import Variable
from dream_utils import Normalize
from dream_augment import DiffAug
from torchvision import transforms
from dream_strategy import NEW_Strategy
import math

class FedDream(FedDistill):
    def __init__(self, args, appr_args, logger):
        super(FedDream, self).__init__(args, appr_args, logger)

    # function that processing the special arguments of current method
    @staticmethod
    def extra_parser(extra_args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--aug_type', type=str, default='color_crop_cutout',
                            help='augmentation strategy for condensation matching')
        parser.add_argument('--mixup_net', type=str, default='cut', choices=('vanilla', 'cut'),
                            help='mixup choice for training networks in condensation stage')
        parser.add_argument('--ipc', type=int, default=1,
                            help='number of condensed data per class')
        parser.add_argument('--batch_real', type=int, default=64,
                            help='batch size for real training data used for matching')
        parser.add_argument('--lr_img', type=float, default=5e-3,
                            help='condensed data learning rate')
        parser.add_argument('--mom_img', type=float, default=0.5,
                            help='condensed data momentum')
        parser.add_argument('--iter', type=int, default=300,
                            help='distilling iterations')
        parser.add_argument('--fix_iter', type=int, default=-1,
                            help='number of outer iteration maintaining the condensation networks')
        parser.add_argument('--factor', type=int, default=2,
                            help='multi-formation factor. (1 for IDC-I)')
        parser.add_argument('--decode_type', type=str, default='single', choices=['single', 'multi', 'bound'],
                            help='multi-formation type')
        parser.add_argument('--bias', type=bool, default=False, 
                            help='match bias or not')
        parser.add_argument('--fc', type=bool, default=False, 
                            help='match fc layer or not')
        parser.add_argument('--interval', type=int, default=10,
                            help='cluster every interval inner loop')
        parser.add_argument('--metric', type=str, default='l1', choices=['mse', 'l1', 'l1_mean', 'l2', 'cos'],
                            help='matching objective')
        parser.add_argument('--inner_loop', type=int, default=100,
                            help='number of inner iteration')

        return parser.parse_args(extra_args)
    
    def remove_aug(self, augtype, remove_aug):
        aug_list = []
        for aug in augtype.split('_'):
            if aug not in remove_aug.split('_'):
                aug_list.append(aug)
        
        return '_'.join(aug_list)

    def diffaug(self):
        """Differentiable augmentation for condensation
        """
        aug_type = self.appr_args.aug_type
        normalize = Normalize(mean=self.args.mean, std=self.args.std, device=self.args.device)
        self.logger.info('Augmentataion Matching: %s' % aug_type)
        augment = DiffAug(strategy=aug_type, batch=True)
        aug_batch = transforms.Compose([normalize, augment])

        if self.appr_args.mixup_net == 'cut':
            aug_type = self.remove_aug(aug_type, 'cutout')
        self.logger.info('Augmentataion Net update: %s' % aug_type)
        augment_rand = DiffAug(strategy=aug_type, batch=False)
        aug_rand = transforms.Compose([normalize, augment_rand])

        return aug_batch, aug_rand
    
    def img_denormalize(self, img):
        """Scaling and shift a batch of images (NCHW)
        """
        mean = self.args.mean
        std = self.args.std
        nch = img.shape[1]

        mean = torch.tensor(mean, device=img.device).reshape(1, nch, 1, 1)
        std = torch.tensor(std, device=img.device).reshape(1, nch, 1, 1)

        return img * std + mean
    
    def save_img(self, save_dir, img, unnormalize=True, max_num=200, size=64, nrow=10):
        img = img[:max_num].detach()
        if unnormalize:
            img = self.img_denormalize(img)
        img = torch.clamp(img, min=0., max=1.)

        if img.shape[-1] > size:
            img = F.interpolate(img, size)
        save_image(img.cpu(), save_dir, nrow=nrow)
    
    def run(self):

        self.appr_args.dsa_param = ParamDiffAug()
        self.appr_args.dsa = False if self.appr_args.dsa_strategy in ['none', 'None'] else True
        if self.args.approach == 'feddc':
            self.appr_args.dsa = False
        elif self.args.approach == 'feddsa':
            self.appr_args.dsa = True
        self.appr_args.outer_loop, self.appr_args.inner_loop = get_loops(self.appr_args.ipc)
        
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

        for round in range(self.args.n_comm_round):
            self.logger.info('Communication Round: %d' % round)

            party_list_this_round = random.sample([_ for _ in range(self.args.C)], 
                                                int(self.args.C * self.args.sample_fraction))
            party_list_this_round.sort()

            for c_idx in party_list_this_round:
                self.logger.info('Client %d' % c_idx)

                train_ds_c = DatasetSplit(train_ds, self.party2dataidx['train'][c_idx])
                val_ds_c = DatasetSplit(val_ds, self.party2dataidx['val'][c_idx])

                train_dl = DataLoader(train_ds_c, num_workers=8, prefetch_factor=16*self.args.train_bs,
                                    batch_size=self.args.train_bs, shuffle=True, drop_last=False, pin_memory=True)
                val_dl = DataLoader(val_ds_c, num_workers=8, prefetch_factor=16*self.args.test_bs,
                                    batch_size=self.args.test_bs, shuffle=False, pin_memory=True)
                self.logger.info('Train batches: %d' % len(train_dl))
                self.logger.info('Val batches: %d' % len(val_dl))

                self.logger.info('Organize the real dataset')
                labels_all = [train_ds.targets[i] for i in self.party2dataidx['train'][c_idx]]
                indices_class = [[] for _ in range(self.args.n_classes)]
                for _, lab in enumerate(labels_all):
                    indices_class[lab].append(_)

                for _ in range(self.args.n_classes):
                    self.logger.info('class c = %d: %d real images' % (_, len(indices_class[_])))

                self.logger.info('Define synthetic data')
                image_syn = torch.randn(size=(self.args.n_classes * self.appr_args.ipc, self.appr_args.channel, 
                                        self.appr_args.im_size[0], self.appr_args.im_size[1]), dtype=torch.float, requires_grad=True)
                label_syn = torch.tensor(np.array([np.ones(self.appr_args.ipc)*i for i in range(self.args.n_classes)]),
                                        dtype=torch.long, requires_grad=False).view(-1)
                image_syn.data = torch.clamp(image_syn.data / 4 + 0.5, min=0., max=1.)
                if self.appr_args.init == 'real':
                    for c in range(self.args.n_classes):
                        image_syn.data[c*self.appr_args.ipc: (c+1)*self.appr_args.ipc] = self.get_images(c, self.appr_args.ipc, indices_class, train_ds_c).detach().data
                
                if self.args.device != 'cpu':
                    net = nn.DataParallel(local_nets[c_idx])
                    net.to(self.args.device)
                net.eval()
                optimizer_net = optim.SGD(net.parameters(), lr=self.args.lr, 
                                          momentum=self.args.momentum, weight_decay=self.args.weight_decay)
                criterion = nn.CrossEntropyLoss()
                aug, aug_rand = self.diffaug(self.args)

                self.logger.info('KMean initialize synset')
                for c in range(self.args.n_classes):
                    indices = indices_class[c][:self.appr_args.ipc]
                    img = torch.stack([train_ds_c[i][0] for i in indices])
                    strategy = NEW_Strategy(img, net)
                    query_idxs = strategy.query(self.appr_args.ipc)
                    image_syn.data[c * self.appr_args.ipc: (c+1) * self.appr_args.ipc] = img.data.to(self.args.device)
                
                self.logger.info('init_size: ', image_syn.data.size())
                save_name = os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'init_client{}_'.format(c_idx)+self.args.log_file_name+'.png')
                self.save_img(save_name, image_syn.data, unnormalize=False)

                self.logger.info('Condense begins...')
                best_img_syn, best_lab_syn = self.DREAM(net, indices_class, image_syn, label_syn, train_ds_c, val_ds_c, aug, aug_rand)

    def decode_zoom(self, img, target, factor):
        """Uniform multi-formation
        """
        h = img.shape[-1]
        remained = h % factor
        if remained > 0:
            img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
        s_crop = math.ceil(h / factor)
        n_crop = factor**2

        cropped = []
        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
        cropped = torch.cat(cropped)
        data_dec = nn.Upsample(size=self.im_size, mode='bilinear')(cropped)
        target_dec = torch.cat([target for _ in range(n_crop)])

        return data_dec, target_dec

    def decode(self, data, target, bound=128):
        """Multi-formation
        """
        if self.appr_args.factor > 1:
            if self.appr_args.decode_type == 'multi':
                data, target = self.decode_zoom_multi(data, target, self.appr_args.factor)
            elif self.appr_args.decode_type == 'bound':
                data, target = self.decode_zoom_bound(data, target, self.appr_args.factor, bound=bound)
            else:
                data, target = self.decode_zoom(data, target, self.appr_args.factor)

        return data, target
    
    def subsample(self, data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            indices = np.random.permutation(data.shape[0])
            data = data[indices[:max_size]]
            target = target[indices[:max_size]]

        return data, target

    def sample(self, c, image_syn, label_syn, max_size=128):
        """Sample synthetic data per class
        """
        idx_from = self.appr_args.ipc * c
        idx_to = self.appr_args.ipc * (c + 1)
        data = image_syn[idx_from:idx_to]
        target = label_syn[idx_from:idx_to]

        data, target = self.decode(data, target, bound=max_size)
        data, target = self.subsample(data, target, max_size=max_size)
        return data, target
    
    def dist(self, x, y, method='mse'):
        """Distance objectives
        """
        if method == 'mse':
            dist_ = (x - y).pow(2).sum()
        elif method == 'l1':
            dist_ = (x - y).abs().sum()
        elif method == 'l1_mean':
            n_b = x.shape[0]
            dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
        elif method == 'cos':
            x = x.reshape(x.shape[0], -1)
            y = y.reshape(y.shape[0], -1)
            dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                            (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))

        return dist_

    def add_loss(self, loss_sum, loss):
        if loss_sum == None:
            return loss
        else:
            return loss_sum + loss
    
    def match_loss(self, img_real, img_syn, lab_real, lab_syn, model):
        """Matching losses (gradient)
        """
        criterion = nn.CrossEntropyLoss()

        output_real = model(img_real)
        loss_real = criterion(output_real, lab_real)
        g_real = torch.autograd.grad(loss_real, model.parameters())
        g_real = list((g.detach() for g in g_real))

        output_syn = model(img_syn)
        loss_syn = criterion(output_syn, lab_syn)
        g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

        for i in range(len(g_real)):
            if (len(g_real[i].shape) == 1) and not self.appr_args.bias:  # bias, normliazation
                continue
            if (len(g_real[i].shape) == 2) and not self.appr_args.fc:
                continue

            loss = self.add_loss(loss, self.dist(g_real[i], g_syn[i], method=self.appr_args.metric))

        return loss

    def DREAM(self, net, indices_class, image_syn, label_syn, train_ds, val_ds, aug, aug_rand):

        optimizer_img = optim.SGD([image_syn, ], lr=self.appr_args.lr_img, momentum=self.appr_args.mom_img)
        self.appr_args.fix_iter = max(1, self.appr_args.fix_iter)
        query_list = torch.tensor(np.ones(shape=(self.args.n_classes, self.appr_args.batch_real)),
                                dtype=torch.long, requires_grad=False, device=self.args.device)

        for it in range(self.appr_args.iter):
            if it % self.appr_args.fix_iter == 0 and it != 0:
                if self.args.dataset in ['mnist', 'cifar10']:
                    model = self.args.model
                    self.args.model = 'ConvNet'
                    net = get_network(self.args)
                    self.args.model = model
                else:
                    net = get_network(self.args)
                if self.args.device != 'cpu':
                    net = nn.DataParallel(net)
                    net.to(self.args.device)
                net.train()
                optimizer_net = optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, 
                                            weight_decay=self.args.weight_decay)
                criterion = nn.CrossEntropyLoss()

            loss_total = 0
            image_syn.data = torch.clamp(image_syn.data, min=0., max=1.)
            
            for il in range(self.appr_args.inner_loop):
                for c in range(self.args.n_classes):

                    indices = indices_class[c]
                    img_c = torch.stack([train_ds[i][0] for i in indices])

                    if il % self.appr_args.interval == 0:
                        strategy = NEW_Strategy(img_c, net)
                        query_idxs = strategy.query(self.appr_args.batch_real)
                        query_list[c] = query_idxs

                    img = img_c[query_list[c]]
                    lab = torch.tensor([np.ones(img.size(0))*c], dtype=torch.long, requires_grad=False, device=self.args.device).view(-1)
                    img_syn, lab_syn = self.sample(c, image_syn, label_syn, max_size=self.appr_args.batch_syn_max)
                    n = img.shape[0]
                    img_aug = aug(torch.cat([img, img_syn]))
                    
                    loss = self.match_loss(img_aug[:n], img_aug[n:], lab, lab_syn, net)
                    loss_total += loss.item()

                    optimizer_img.zero_grad()
                    loss.backward()
                    optimizer_img.step()

                train_epoch(net, train_ds, criterion, optimizer_net, aug=aug_rand)

            if it % 10 == 0:
                self.logger.info('Iter: %03d loss: %.3f' % (it, loss_total / self.args.n_classes / self.appr_args.inner_loop))
            