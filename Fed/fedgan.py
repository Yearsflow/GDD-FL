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

class FedGAN(FedDistill):
    def __init__(self, args, appr_args, logger):
        super(FedGAN, self).__init__(args, appr_args, logger)

    # function that processing the special arguments of current method
    @staticmethod
    def extra_parser(extra_args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--diffaug_choice', type=str, default='Auto',
                            help='DSA augmentation choice')

        return parser.parse_args(extra_args)
    
    def run(self):

        if self.appr_args.diffaug_choice == 'Auto':
            if self.args.dataset == 'mnist':
                self.appr_args.diffaug_choice = 'color_crop_cutout_scale_rotate'
            else:
                self.appr_args.diffaug_choice = 'color_crop_cutout_flip_scale_rotate'
        else:
            self.appr_args.diffaug_choice = 'None'
        
        