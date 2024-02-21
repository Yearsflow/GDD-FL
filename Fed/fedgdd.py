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

class FedGDD(FedDistill):
    def __init__(self, args, appr_args, logger):
        super(FedGDD, self).__init__(args, appr_args, logger)

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
    
    