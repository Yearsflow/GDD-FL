from utils.common_utils import get_dataloader, DatasetSplit, get_network
from networks import ResNet18
import torch
import os
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from .central import Central

class RS(Central):

    def __init__(self, args, appr_args, logger):
        super(RS, self).__init__(args, appr_args, logger)

    @staticmethod
    def extra_parser(extra_args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--n_per_class', type=int, default=10,
                            help='how many samples randomly selected for training')

        return parser.parse_args(extra_args)
    
    def run(self):

        train_ds, val_ds, public_ds, test_ds, n_per_class = get_dataloader(self.args)

        if self.args.dataset in {'mnist', 'cifar10'}:
            n_classes = 10
        elif self.args.dataset == 'cifar100':
            n_classes = 100
        elif self.args.dataset == 'isic2020':
            n_classes = 2
        elif self.args.dataset == 'EyePACS':
            n_classes = 5
        else:
            raise NotImplementedError('Dataset Not Supported')

        num_per_class = [0 for _ in range(n_classes)]
        dataidxs = []
        targets = train_ds.targets
        for i in range(len(targets)):
            if num_per_class[targets[i]] < self.appr_args.n_per_class:
                num_per_class[targets[i]] += 1
                dataidxs.append(i)
        train_dl = DataLoader(DatasetSplit(train_ds, dataidxs), num_workers=8, shuffle=True, prefetch_factor=2*64,
                            batch_size=self.args.train_bs, drop_last=False, pin_memory=True)
        val_dl = DataLoader(dataset=val_ds, batch_size=self.args.test_bs, num_workers=8, 
                            prefetch_factor=2*64, shuffle=False, pin_memory=True)
        test_dl = DataLoader(dataset=test_ds, batch_size=self.args.test_bs, num_workers=8,
                            prefetch_factor=2*64, shuffle=False, pin_memory=True)

        net = get_network(self.args)
        if self.args.device != 'cpu':
            net = nn.DataParallel(net)
        net.to(self.args.device)
        
        losses, accs, w = self.train(net, train_dl, val_dl)
        
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