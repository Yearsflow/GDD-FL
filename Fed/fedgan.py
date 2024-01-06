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
from utils.common_utils import get_dataloader, DatasetSplit, get_network, get_loops, ParamDiffAug, DiffAugment, match_loss, augment
from utils.fedgan_utils import Normalize, rand_bbox, AverageMeter
from networks import Generator, Discriminator
from torch.autograd import Variable
from utils.fedgan_augment import DiffAug
from torchvision import transforms

class FedGAN(FedDistill):
    def __init__(self, args, appr_args, logger):
        super(FedGAN, self).__init__(args, appr_args, logger)

    # function that processing the special arguments of current method
    @staticmethod
    def extra_parser(extra_args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--mix_p', type=float, default=-1.0)
        parser.add_argument('--beta', type=float, default=1.0)
        parser.add_argument('--aug_type', type=str, default='color_crop_cutout')
        parser.add_argument('--mixup_net', type=str, default='cut')
        parser.add_argument('--eval_bs', type=int, default=64,
                            help='batch size for validation')
        parser.add_argument('--eval_epochs', type=int, default=100,
                            help='training epochs for validation step')
        parser.add_argument('--eval_lr', type=float, default=0.01,
                            help='learning rate of training network for validation step')
        parser.add_argument('--eval_mom', type=float, default=0.9,
                            help='momentum for optimizer in validation step')
        parser.add_argument('--eval_wd', type=float, default=1e-5,
                            help='weight decay for optimizer in validation step')
        parser.add_argument('--GAN_lr', type=float, default=1e-4,
                            help='learning rate for training GAN')
        parser.add_argument('--GAN_epochs', type=int, default=300,
                            help='number of epochs for training GAN')
        parser.add_argument('--GAN_bs', type=int, default=64,
                            help='batch size for training GAN')
        parser.add_argument('--dim', type=int, default=100,
                            help='number of noise dimension')
        parser.add_argument('--ipc', type=int, default=1,
                            help='number of distilled images')

        return parser.parse_args(extra_args)
    
    def remove_aug(self, augtype, remove_aug):
        aug_list = []
        for aug in augtype.split("_"):
            if aug not in remove_aug.split("_"):
                aug_list.append(aug)

        return "_".join(aug_list)
    
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
    
    def calc_gradient_penalty(self, args, discriminator, img_real, img_syn):
        ''' Gradient penalty from Wasserstein GAN
        '''
        LAMBDA = 10
        n_size = img_real.shape[-1]
        batch_size = img_real.shape[0]
        n_channels = img_real.shape[1]

        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, int(img_real.nelement() / batch_size)).contiguous()
        alpha = alpha.view(batch_size, n_channels, n_size, n_size)
        alpha = alpha.cuda()

        img_syn = img_syn.view(batch_size, n_channels, n_size, n_size)
        interpolates = alpha * img_real.detach() + ((1 - alpha) * img_syn.detach())

        interpolates = interpolates.cuda()
        interpolates.requires_grad_(True)

        disc_interpolates, _ = discriminator(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty
    
    def train_epoch(self, args, G, D, optim_G, optim_D, trainloader, criterion, aug, aug_rand):
        ''' The main training function for the generator '''
        G.train()
        gen_losses = AverageMeter()
        disc_losses = AverageMeter()

        for batch_idx, (x, target) in enumerate(trainloader):
            x = x.to(self.args.device, non_blocking=True)
            target = target.long().to(self.args.device, non_blocking=True)

            # train the generator
            D.eval()
            optim_G.zero_grad()

            # obtain the noise with one-hot class labels
            if len(target) == self.appr_args.GAN_bs:
                noise = torch.normal(0, 1, (self.appr_args.GAN_bs, self.appr_args.dim))
                lab_onehot = torch.zeros((self.appr_args.GAN_bs, self.args.n_classes))
                lab_onehot[torch.arange(self.appr_args.GAN_bs), target] = 1
                noise[torch.arange(self.appr_args.GAN_bs), :self.args.n_classes] = lab_onehot[torch.arange(self.appr_args.GAN_bs)]
            else:
                noise = torch.normal(0, 1, (len(target), self.appr_args.dim))
                lab_onehot = torch.zeros((len(target), self.args.n_classes))
                lab_onehot[torch.arange(len(target)), target] = 1
                noise[torch.arange(len(target)), :self.args.n_classes] = lab_onehot[torch.arange(len(target))]
            noise = noise.to(self.args.device, non_blocking=True)

            img_syn = G(noise)
            gen_source, gen_class = D(img_syn)
            gen_source = gen_source.mean()
            gen_class = criterion(gen_class, target)
            gen_loss = - gen_source + gen_class

            gen_loss.backward()
            optim_G.step()

            # train the discriminator
            D.train()
            optim_D.zero_grad()
            if len(target) == self.appr_args.GAN_bs:
                lab_syn = torch.randint(self.args.n_classes, (self.appr_args.GAN_bs,))
                noise = torch.normal(0, 1, (self.appr_args.GAN_bs, self.appr_args.dim))
                lab_onehot = torch.zeros((self.appr_args.GAN_bs, self.args.n_classes))
                lab_onehot[torch.arange(self.appr_args.GAN_bs), target] = 1
                noise[torch.arange(self.appr_args.GAN_bs), :self.args.n_classes] = lab_onehot[torch.arange(self.appr_args.GAN_bs)]
            else:
                lab_syn = torch.randint(self.args.n_classes, (len(target),))
                noise = torch.normal(0, 1, (len(target), self.appr_args.dim))
                lab_onehot = torch.zeros((len(target), self.args.n_classes))
                lab_onehot[torch.arange(len(target)), target] = 1
                noise[torch.arange(len(target)), :self.args.n_classes] = lab_onehot[torch.arange(len(target))]
            noise = noise.to(self.args.device, non_blocking=True)
            lab_syn = lab_syn.to(self.args.device, non_blocking=True)

            with torch.no_grad():
                img_syn = G(noise)
            
            disc_fake_source, disc_fake_class = D(img_syn)
            disc_fake_source = disc_fake_source.mean()
            disc_fake_class = criterion(disc_fake_class, lab_syn)

            disc_real_source, disc_real_class = D(x)
            disc_real_source = disc_real_source.mean()
            disc_real_class = criterion(disc_real_class, target)

            gradient_penalty = self.calc_gradient_penalty(args, D, x, img_syn)

            disc_loss = disc_fake_source - disc_real_source + disc_fake_class + disc_real_class + gradient_penalty
            disc_loss.backward()
            optim_D.step()

            gen_losses.update(gen_loss.item())
            disc_losses.update(disc_loss.item())

        return gen_losses.avg, disc_losses.avg
                        
    def validate(self, args, G, val_dl, criterion, aug_rand):
        ''' Validate the generator performance '''
        net = get_network(self.args)
        if args.device != 'cpu':
            net = nn.DataParallel(net)
            net.to(args.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.appr_args.eval_lr, momentum=self.appr_args.eval_mom,
                              weight_decay=self.appr_args.eval_wd)

        G.eval()
        losses = AverageMeter()
        
        for ep in range(self.appr_args.eval_epochs):
            for batch_idx in range(10 * self.appr_args.ipc // self.appr_args.eval_bs + 1):
                # obtain pseudo samples with the generator
                lab_syn = torch.randint(args.n_classes, (self.appr_args.eval_bs,))
                noise = torch.normal(0, 1, (self.appr_args.eval_bs, self.appr_args.dim))
                lab_onehot = torch.zeros((self.appr_args.eval_bs, args.n_classes))
                lab_onehot[torch.arange(self.appr_args.eval_bs), lab_syn] = 1
                noise[torch.arange(self.appr_args.eval_bs), :args.n_classes] = lab_onehot[torch.arange(self.appr_args.eval_bs)]
                noise = noise.to(args.device, non_blocking=True)
                lab_syn = lab_syn.to(args.device, non_blocking=True)

                with torch.no_grad():
                    img_syn = G(noise)
                    img_syn = aug_rand((img_syn + 1.0) / 2.0)
                
                if np.random.rand(1) < self.appr_args.mix_p and self.appr_args.mixup_net == 'cut':
                    lam = np.random.beta(self.appr_args.beta, self.appr_args.beta)
                    rand_index = torch.randperm(len(img_syn)).cuda()

                    lab_syn_b = lab_syn[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(img_syn.size(), lam)
                    img_syn[:, :, bbx1:bbx2, bby1:bby2] = img_syn[rand_index, :, bbx1:bbx2, bby1:bby2]
                    ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_syn.size()[-1] * img_syn.size()[-2]))

                    output = net(img_syn)
                    loss = criterion(output, lab_syn) * ratio + criterion(output, lab_syn_b) * (1. - ratio)
                else:
                    output = net(img_syn)
                    loss = criterion(output, lab_syn)

                losses.update(loss.item(), img_syn.shape[0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.eval(net, val_dl)    

    def train_GAN(self, G, D, train_dl, val_dl):

        optimizer_G = optim.Adam(G.parameters(), lr=self.appr_args.GAN_lr, betas=(0, 0.9))
        optimizer_D = optim.Adam(D.parameters(), lr=self.appr_args.GAN_lr, betas=(0, 0.9))
        criterion = nn.CrossEntropyLoss()

        best_metric = 0.0
        best_G = None

        aug, aug_rand = self.diffaug()
        for ep in range(self.appr_args.GAN_epochs):
            G.train()
            D.train()
            G_loss, D_loss = self.train_epoch(self.args, G, D, optimizer_G, optimizer_D, train_dl, criterion, aug, aug_rand)

            metric = self.validate(self.args, G, val_dl, criterion, aug_rand)
            if self.args.dataset in ['mnist', 'cifar10']:
                self.logger.info('Epoch: %d G loss: %f D loss: %f Val Acc: %f' % (ep, G_loss, D_loss, metric))
            else:
                self.logger.info('Epoch: %d G loss: %f D loss: %f Val AUC: %f' % (ep, G_loss, D_loss, metric))
            if metric > best_metric:
                best_metric = metric
                best_G = copy.deepcopy(G.state_dict())
        
        return best_G
    
    def get_synthetic_data(self, Gs):

        syn_data = {
            'images': [],
            'label': []
        }
        n = self.appr_args.ipc * self.args.n_classes

        for G in Gs:
            G.eval()
            lab_syn = torch.tensor(np.array([np.ones(self.appr_args.ipc)*i for i in range(self.args.n_classes)]),
                                dtype=torch.long, requires_grad=False).view(-1)
            noise = torch.normal(0, 1, (n, self.appr_args.dim))
            lab_onehot = torch.zeros((n, self.args.n_classes))
            lab_onehot[torch.arange(n), lab_syn] = 1
            noise[torch.arange(n), :self.args.n_classes] = lab_onehot[torch.arange(n)]
            noise = noise.to(self.args.device, non_blocking=True)
            lab_syn = lab_syn.to(self.args.device, non_blocking=True)
            
            with torch.no_grad():
                img_syn = G(noise)

            for j in range(len(img_syn)):
                syn_data['images'].append(img_syn[j].detach().cpu())
                syn_data['label'].append(lab_syn[j].detach().cpu())

        return syn_data

    def run(self):
        
        self.logger.info('Partitioning data...')

        train_ds, val_ds, test_ds, num_per_class = get_dataloader(self.args, request='dataset')
        self.party2dataidx = self.partition(train_ds, val_ds)

        self.logger.info('Training begins...')

        for round in range(self.args.n_comm_round):
            self.logger.info('Communication Round: %d' % round)
            party_list_this_round = random.sample([_ for _ in range(self.args.C)], 
                                                int(self.args.C * self.args.sample_fraction))
            party_list_this_round.sort()

            local_Gs = []

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
                
                self.logger.info('Organize the real dataset')
                labels_all = [train_ds.targets[i] for i in self.party2dataidx['train'][client_idx]]
                indices_class = [[] for _ in range(self.args.n_classes)]
                for _, lab in enumerate(labels_all):
                    indices_class[lab].append(_)
                for _ in range(self.args.n_classes):
                    self.logger.info('class c = %d: %d real images' % (_, len(indices_class[_])))

                self.logger.info('Train GAN')
                local_G = Generator(self.args)
                local_D = Discriminator(self.args)
                if self.args.device != 'cpu':
                    local_G = nn.DataParallel(local_G)
                    local_D = nn.DataParallel(local_D)
                    local_G.to(self.args.device)
                    local_D.to(self.args.device)

                local_w = self.train_GAN(local_G, local_D, train_dl, val_dl)
                local_G.load_state_dict(local_w)
                local_Gs.append(local_G)

                torch.save(local_G.state_dict(),
                    os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'local_client{}_round{}_Generator_'.format(client_idx, round)+self.args.log_file_name+'.pth'))

            self.logger.info('Initialize global net')
            global_net = get_network(self.args)
            if self.args.device != 'cpu':
                global_net = nn.DataParallel(global_net)
                global_net.to(self.args.device)
            
            self.logger.info('Get synthetic data')
            syn_data = self.get_synthetic_data(local_Gs)

            self.logger.info('Train global net')
            global_w = self.global_train(global_net, syn_data)
            global_net.load_state_dict(global_w)
            torch.save(global_w,
                os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'global_{}_round{}_'.format(self.args.model, round)+self.args.log_file_name+'.pth'))

            self.logger.info('Evaluate on test dataset')
            test_dl = DataLoader(dataset=test_ds, batch_size=self.args.test_bs, shuffle=False,
                                num_workers=8, pin_memory=True, prefetch_factor=16*self.args.test_bs)
            
            if self.args.dataset in ['isic2020', 'EyePACS']:
                test_auc = self.eval(global_net, test_dl)
                self.logger.info('>>> Global Model Test AUC: %f' % test_auc)
            else:
                test_acc = self.eval(global_net, test_dl)
                self.logger.info('>>> Global Model Test Accuracy: %f' % test_acc)
