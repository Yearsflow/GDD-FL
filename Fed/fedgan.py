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
from utils import get_dataloader, DatasetSplit, get_network, get_loops, ParamDiffAug, DiffAugment, match_loss, augment
from networks import Generator, Discriminator
from torch.autograd import Variable

class FedGAN(FedDistill):
    def __init__(self, args, appr_args, logger):
        super(FedGAN, self).__init__(args, appr_args, logger)

    # function that processing the special arguments of current method
    @staticmethod
    def extra_parser(extra_args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--diffaug_choice', type=str, default='Auto',
                            help='DSA augmentation choice')
        parser.add_argument('--pt_lr', type=float, default=0.01,
                            help='learning rate for pre-training network')
        parser.add_argument('--pt_optim', type=str, default='sgd', 
                            help='optimizer for pre-training network')
        parser.add_argument('--pt_epochs', type=int, default=100,
                            help='number of epochs for pre-training network')
        parser.add_argument('--pt_bs', type=int, default=64,
                            help='batch size for pre-training network')
        parser.add_argument('--pt_momentum', type=float, default=0.9,
                            help='momentum for pre-training network')
        parser.add_argument('--pt_decay', type=float, default=1e-5,
                            help='weight decay for pre-training network')
        parser.add_argument('--GAN_lr', type=float, default=0.00005,
                            help='learning rate for training WGAN')
        parser.add_argument('--GAN_epochs', type=int, default=2000,
                            help='number of epochs for training WGAN')
        parser.add_argument('--n_critic', type=int, default=5, 
                            help='number of training steps for discriminator per iter')
        parser.add_argument('--clip_value', type=float, default=0.01,
                            help='lower and upper clip value for disc. weights')
        parser.add_argument('--z_lr', type=float, default=0.1,
                            help='fixed learning rate for training z')
        parser.add_argument('--z_epochs', type=int, default=300, 
                            help='epochs to train z')
        parser.add_argument('--z_bs', type=int, default=64,
                            help='batch size for training z')
        parser.add_argument('--ratio_div', type=float, default=0.001,
                            help='ratio_div')
        parser.add_argument('--condense_bs', type=int, default=64,
                            help='batch size for sampling real images to condense knowledge')
        parser.add_argument('--ipc', type=int, default=1,
                            help='number of distilled images')

        return parser.parse_args(extra_args)

    def local_pretrain(self, net, train_dl, val_dl):

        criterion = nn.CrossEntropyLoss()
        if self.appr_args.pt_optim == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=self.appr_args.pt_lr, 
                                  momentum=self.appr_args.pt_momentum,
                                  weight_decay=self.appr_args.pt_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

        best_metric, best_w = 0, None

        for ep in range(self.appr_args.pt_epochs):
            epoch_loss_collector = []
            total, correct = 0, 0
            logits, labels = [], []

            net.train()
            for batch_idx, (x, target) in enumerate(train_dl):
                x = x.to(self.args.device, non_blocking=True)
                x = DiffAugment(x, self.appr_args.diffaug_choice, seed=self.args.init_seed, param=ParamDiffAug())
                target = target.long().to(self.args.device, non_blocking=True)
                optimizer.zero_grad()
                out = net(x)

                loss = criterion(out, target)
                _, pred_label = torch.max(out, 1)
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()
                _, pred_label = torch.max(out.data, 1)
                if self.args.n_classes > 2:
                    for i in range(len(out)):
                            logits.append(F.softmax(out[i], dim=0).cpu().tolist())
                else:
                    for i in range(len(out)):
                        logits.append(F.softmax(out[i], dim=0).cpu().tolist()[1])
                labels += target.cpu().tolist()
                loss.backward()
                optimizer.step()

                epoch_loss_collector.append(loss.item())

            epoch_train_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_train_acc = correct / float(total)
            if self.args.dataset in ['isic2020', 'EyePACS']:
                if self.args.n_classes > 2:
                    epoch_train_auc = roc_auc_score(np.array(labels), np.array(logits), multi_class='ovo', labels=[_ for _ in range(self.args.n_classes)])
                else:
                    epoch_train_auc = roc_auc_score(np.array(labels), np.array(logits))
            
            epoch_loss_collector = []
            logits, labels = [], []
            total, correct = 0, 0

            net.eval()
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(val_dl):
                    x = x.to(self.args.device, non_blocking=True)
                    target = target.long().to(self.args.device, non_blocking=True)
                    out = net(x)

                    loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    if self.args.n_classes > 2:
                        for i in range(len(out)):
                            logits.append(F.softmax(out[i], dim=0).cpu().tolist())
                    else:
                        for i in range(len(out)):
                            logits.append(F.softmax(out[i], dim=0).cpu().tolist()[1])
                    labels += target.data.cpu().tolist()
                    epoch_loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

            epoch_val_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_val_acc = correct / float(total)
            if self.args.dataset in ['isic2020', 'EyePACS']:
                if self.args.n_classes > 2:
                    epoch_val_auc = roc_auc_score(np.array(labels), np.array(logits), multi_class='ovo', labels=[_ for _ in range(self.args.n_classes)])
                else:
                    epoch_val_auc = roc_auc_score(np.array(labels), np.array(logits))
                
            if self.args.dataset in ['isic2020', 'EyePACS']:
                scheduler.step(epoch_val_auc)
                if epoch_val_auc > best_metric:
                    best_metric = epoch_val_auc
                    best_w = copy.deepcopy(net.state_dict())
            else:
                scheduler.step(epoch_val_acc)
                if epoch_val_acc > best_metric:
                    best_metric = epoch_val_acc
                    best_w = copy.deepcopy(net.state_dict())

            if self.args.dataset in ['isic2020', 'EyePACS']:
                self.logger.info('Epoch: %d Train loss: %f Train Acc: %f Train AUC: %f Val loss: %f Val Acc: %f Val AUC: %f' %
                                (ep, epoch_train_loss, epoch_train_acc, epoch_train_auc, epoch_val_loss, epoch_val_acc, epoch_val_auc))
            else:
                self.logger.info('Epoch: %d Train loss: %f Train Acc: %f Val loss: %f Val Acc: %f' %
                                (ep, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc))

        return best_w
    
    def train_WGAN(self, G, D, train_dl, latent_dim=128):

        optimizer_G = optim.RMSprop(G.parameters(), lr=self.appr_args.GAN_lr)
        optimizer_D = optim.RMSprop(D.parameters(), lr=self.appr_args.GAN_lr)

        if self.args.device != 'cpu':
            G = nn.DataParallel(G)
            D = nn.DataParallel(D)
            G.to(self.args.device)
            D.to(self.args.device)

        for ep in range(self.appr_args.GAN_epochs):
            for batch_idx, (x, target) in enumerate(train_dl):
                real_imgs = Variable(x.to(self.args.device))
                optimizer_D.zero_grad()

                z = Variable(torch.FloatTensor(np.random.normal(0, 1, (x.shape[0], latent_dim))).to(self.args.device))

                fake_imgs = G(z).detach()
                loss_D = -torch.mean(D(real_imgs)) + torch.mean(D(fake_imgs))

                loss_D.backward()
                optimizer_D.step()

                for p in D.parameters():
                    p.data.clamp_(-self.appr_args.clip_value, self.appr_args.clip_value)

                if batch_idx % self.appr_args.n_critic == 0:

                    optimizer_G.zero_grad()
                    gen_imgs = G(z)
                    loss_G = -torch.mean(D(gen_imgs))

                    loss_G.backward()
                    optimizer_G.step()
            
                    self.logger.info('Epoch: %d Batch: %d D loss: %f G loss: %f' % (ep, batch_idx, loss_D.item(), loss_G.item()))

    def get_images(self, c, n, indices_class, train_ds):

        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        images = []
        for idx in idx_shuffle:
            images.append(torch.unsqueeze(train_ds[idx][0], dim=0))
        return torch.cat(images, dim=0).to(self.args.device)

    def train_z(self, net, G, indices_class, train_ds):

        dim_z = 128

        z = []
        optimizers = []
        for c in range(self.args.n_classes):
            idxs = indices_class[c]
            z_c = torch.randn(size=(len(idxs), dim_z), dtype=torch.float, requires_grad=True)
            z.append(z_c)
            optimizer_c = optim.Adam([z_c], lr=self.appr_args.z_lr, betas=[0.9, 0.999])
            optimizers.append(optimizer_c)

        self.logger.info('Train WGAN Inversion')
        
        for ep in range(self.appr_args.z_epochs):    
            loss_feat_all = []
            loss_pixel_all = []
            loss_all = []

            for c in range(self.args.n_classes):
                idxs = indices_class[c]
                labs = [c for _ in range(len(idxs))]

                net.train()
                for param in net.parameters():
                    param.requires_grad = False
                embed = net.module.embed
                
                for batch_idx in range(int(np.ceil(len(idxs) // self.appr_args.z_bs))):
                    idx = np.arange(batch_idx * self.appr_args.z_bs, (batch_idx + 1) * self.appr_args.z_bs)
                    z_syn = z[c][idx].to(self.args.device, non_blocking=True)
                    lab_syn = torch.tensor([labs[i] for i in idx], dtype=torch.long, device=self.args.device)
                    img_syn = G(z_syn)
                    feat_syn = embed(img_syn)

                    images = []
                    for i in idx:
                        images.append(torch.unsqueeze(train_ds[i][0], dim=0))
                    img_real = torch.cat(images, dim=0).to(self.args.device, non_blocking=True).detach()
                    feat_real = embed(img_real).detach()

                    loss_feat = torch.mean((feat_syn - feat_real) ** 2)
                    loss_pixel = torch.mean((img_syn - img_real) ** 2)
                    loss = loss_feat + loss_pixel

                    optimizers[c].zero_grad()
                    loss.backward()
                    optimizers[c].step()

                    loss_feat_all.append(loss_feat.item())
                    loss_pixel_all.append(loss_pixel.item())
                    loss_all.append(loss.item())

            self.logger.info('Epoch: %d loss: feat = %.4f pixel = %.4f sum = %.4f' % 
                             (ep, np.mean(loss_feat_all), np.mean(loss_pixel_all), np.mean(loss_all)))
        
        optimizers = []
        for c in range(self.args.n_classes):
            optimizer_c = optim.Adam([z[c]], lr=self.appr_args.z_lr, betas=[0.9, 0.999])
            optimizers.append(optimizer_c)

        self.logger.info('Train WGAN z')

        for ep in range(self.appr_args.z_epochs):
            loss_divs_all = []
            loss_cond_all = []
            loss_all = []

            for c in range(self.args.n_classes):
                idxs = indices_class[c]
                labs = [c for _ in range(len(idxs))]

                net.train()
                for param in net.parameters():
                    param.requires_grad = False
                embed = net.module.embed

                for batch_idx in range(int(np.ceil(len(idxs) // self.appr_args.z_bs))):
                    idx = np.arange(batch_idx * self.appr_args.z_bs, (batch_idx + 1) * self.appr_args.z_bs)
                    z_syn = z[c][idx].to(self.args.device, non_blocking=True)
                    lab_syn = torch.tensor([labs[i] for i in idx], dtype=torch.long, device=self.args.device)
                    img_syn = G(z_syn)
                    img_syn = DiffAugment(img_syn, self.appr_args.diffaug_choice, seed=self.args.init_seed, param=ParamDiffAug())
                    feat_syn = embed(img_syn)

                    images = []
                    for i in idx:
                        images.append(torch.unsqueeze(train_ds[i][0], dim=0))
                    img_real = torch.cat(images, dim=0).to(self.args.device, non_blocking=True).detach()
                    feat_real = embed(img_real).detach()

                    if self.appr_args.ratio_div > 0.00001:
                        loss_divs = torch.mean(torch.sum((feat_syn - feat_real) ** 2, dim=-1))
                    else:
                        loss_divs = torch.tensor(0.0).to(self.args.device)
                    img_cond = self.get_images(c, self.appr_args.condense_bs, indices_class, train_ds)
                    img_cond = DiffAugment(img_cond, self.appr_args.diffaug_choice, seed=self.args.init_seed, param=ParamDiffAug())

                    feat_cond = torch.mean(embed(img_cond).detach(), dim=0)
                    loss_cond = torch.sum((torch.mean(feat_syn, dim=0) - feat_cond) ** 2)

                    loss = self.appr_args.ratio_div * loss_divs + (1 - self.appr_args.ratio_div) * loss_cond

                    optimizers[c].zero_grad()
                    loss.backward()
                    optimizers[c].step()

                    loss_divs_all.append(loss_feat.item())
                    loss_cond_all.append(loss_pixel.item())
                    loss_all.append(loss.item())

            self.logger.info('Epoch: %d loss: divs = %.4f * %.3f cond = %.4f * %.3f weighted_sum = %.4f' % 
                             (ep, np.mean(loss_divs_all), self.appr_args.ratio_div, np.mean(loss_cond_all), (1 - self.appr_args.ratio_div), np.mean(loss_all)))

        return z
    
    def renormalize(self, img):

        mean_GAN = [0.5, 0.5, 0.5]
        std_GAN = [0.5, 0.5, 0.5]
        
        return torch.cat([(((img[:, 0] * std_GAN[0] + mean_GAN[0]) - self.args.mean[0]) / self.args.std[0]).unsqueeze(1),
                          (((img[:, 1] * std_GAN[1] + mean_GAN[1]) - self.args.mean[1]) / self.args.std[1]).unsqueeze(1),
                          (((img[:, 2] * std_GAN[2] + mean_GAN[2]) - self.args.mean[2]) / self.args.std[2]).unsqueeze(1)], dim=1)
    
    def get_synthetic_data(self, G, z):

        syn_data = {
            'images': [],
            'label': []
        }

        for i in range(len(z)):
            for c in range(self.args.n_classes):
                z_c = torch.tensor(z[i][c][:self.appr_args.ipc], device=self.args.device).detach()
                lab_c = torch.tensor([c for _ in range(self.appr_args.ipc)], dtype=torch.long)
                img_syn = copy.deepcopy(G(z_c))
                img_syn = self.renormalize(img_syn)
                img_syn = img_syn.view(self.args.channel, self.args.im_size[0], self.args.im_size[1])
                for j in range(len(img_syn)):
                    syn_data['images'].append(img_syn[j].detach().cpu())
                for j in range(len(lab_c)):
                    syn_data['label'].append(lab_c[j])

        return syn_data

    def run(self):

        if self.appr_args.diffaug_choice == 'Auto':
            if self.args.dataset == 'mnist':
                self.appr_args.diffaug_choice = 'color_crop_cutout_scale_rotate'
            else:
                self.appr_args.diffaug_choice = 'color_crop_cutout_flip_scale_rotate'
        else:
            self.appr_args.diffaug_choice = 'None'
        
        self.logger.info('Partitioning data...')

        train_ds, val_ds, test_ds, num_per_class = get_dataloader(self.args, request='dataset')
        self.party2dataidx = self.partition(train_ds, val_ds)

        self.logger.info('Training begins...')

        for round in range(self.args.n_comm_round):
            self.logger.info('Communication Round: %d' % round)
            party_list_this_round = random.sample([_ for _ in range(self.args.C)], 
                                                int(self.args.C * self.args.sample_fraction))
            party_list_this_round.sort()

            client_z = []

            for client_idx in party_list_this_round:
                self.logger.info('Client %d' % client_idx)

                train_ds_c = DatasetSplit(train_ds, self.party2dataidx['train'][client_idx])
                val_ds_c = DatasetSplit(val_ds, self.party2dataidx['val'][client_idx])
                train_dl = DataLoader(train_ds_c, num_workers=8, prefetch_factor=16*self.args.train_bs,
                                    batch_size=self.args.train_bs, shuffle=True, drop_last=False, pin_memory=True)
                val_dl = DataLoader(val_ds_c, num_workers=8, prefetch_factor=16*self.args.test_bs,
                                    batch_size=self.args.test_bs, shuffle=False, pin_memory=True)
                
                self.logger.info('Train batches: %d' % len(train_dl))
                self.logger.info('Val batches: %d' % len(val_dl))

                self.logger.info('Initialize local feature extractor')
                if self.args.dataset in ['mnist', 'cifar10']:
                    model = self.args.model
                    self.args.model = 'ConvNet'
                    local_net = get_network(self.args)
                    self.args.model = model
                else:
                    local_net = get_network(self.args)
                if self.args.device != 'cpu':
                    local_net = nn.DataParallel(local_net)
                    local_net.to(self.args.device)
                
                self.logger.info('Organize the real dataset')
                labels_all = [train_ds.targets[i] for i in self.party2dataidx['train'][client_idx]]
                indices_class = [[] for _ in range(self.args.n_classes)]
                for _, lab in enumerate(labels_all):
                    indices_class[lab].append(_)
                for _ in range(self.args.n_classes):
                    self.logger.info('class c = %d: %d real images' % (_, len(indices_class[_])))

                self.logger.info('Pretrain feature extractor')
                w = self.local_pretrain(local_net, train_dl, val_dl)
                local_net.load_state_dict(w)
                torch.save(w,
                    os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'local_client{}_round{}_FeatureExtractor_'.format(client_idx, round)+self.args.log_file_name+'.pth'))

                self.logger.info('Train WGAN')
                img_shape = (self.args.channel, self.args.im_size[0], self.args.im_size[1])
                local_G = Generator(img_shape=img_shape)
                local_D = Discriminator(img_shape=img_shape)

                self.train_WGAN(local_G, local_D, train_dl)
                local_G.eval()
                for param in local_G.parameters():
                    param.requires_grad = False

                torch.save(local_G.state_dict(),
                    os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'local_client{}_round{}_WGAN_Generator_'.format(client_idx, round)+self.args.log_file_name+'.pth'))

                self.logger.info('Train z')
                z = self.train_z(local_net, local_G, indices_class, train_ds_c)
                z_mean = np.mean([torch.mean(z[c]).item() for c in range(self.args.n_classes)])
                z_std = np.mean([torch.std(z[c].reshape((-1))).item() for c in range(self.args.n_classes)])
                self.logger.info('z mean = %.4f, z std = %.4f' % (z_mean, z_std))

                torch.save(z,
                    os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'local_client{}_round{}_ITGAN_z_'.format(client_idx, round)+self.args.log_file_name+'.pth'))
                client_z.append(z)
            
            self.logger.info('Initialize global net')
            global_net = get_network(self.args)
            if self.args.device != 'cpu':
                global_net = nn.DataParallel(global_net)
                global_net.to(self.args.device)
            
            self.logger.info('Get synthetic dataset')
            syn_data = self.get_synthetic_data(local_G, client_z)
            
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
