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
        if self.args.dataset == 'isic2020':
            self.appr_args.n_classes = 2
            self.appr_args.channel = 3
            self.appr_args.im_size = (224, 224)
            self.appr_args.mean = [0.485, 0.456, 0.406]
            self.appr_args.std = [0.229, 0.224, 0.225]
        elif self.args.dataset == 'EyePACS':
            self.appr_args.n_classes = 5
            self.appr_args.channel = 3
            self.appr_args.im_size = (224, 224)
            self.appr_args.mean = [0.485, 0.456, 0.406]
            self.appr_args.std = [0.229, 0.224, 0.225]
        elif self.args.dataset == 'mnist':
            self.appr_args.n_classes = 10
            self.appr_args.channel = 1
            self.appr_args.im_size = (28, 28)
            self.appr_args.mean = [0.1307]
            self.appr_args.std = [0.3081]
        elif self.args.dataset == 'cifar10':
            self.appr_args.n_classes = 10
            self.appr_args.channel = 3
            self.appr_args.im_size = (32, 32)
            self.appr_args.mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
            self.appr_args.std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        else:
            raise NotImplementedError('Dataset Not Supported')
        
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
                
                query_list = torch.tensor(np.ones(shape=(self.args.n_classes, self.appr_args.batch_real)),
                                          dtype=torch.long, requires_grad=False, device=self.args.device)
                self.logger.info('init_size: ', image_syn.data.size())
                save_name = os.path.join(self.args.ckptdir, self.args.mode, self.args.approach, 'init_client{}_'.format(c_idx)+self.args.log_file_name+'.png')
                self.save_img(save_name, image_syn.data, unnormalize=False)

                self.logger.info('Condense begins...')
                best_img_syn, best_lab_syn = self.DREAM(net, indices_class, image_syn, label_syn, train_ds_c, val_ds_c)


    def DREAM(self, net, indices_class, image_syn, label_syn, train_ds, val_ds):

        optimizer_img = optim.SGD([image_syn, ], lr=self.appr_args.lr_img, momentum=self.appr_args.mom_img)
        