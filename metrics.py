import argparse
import torch
from sklearn.metrics import classification_report
import os
from utils.common_utils import get_dataloader, get_network, get_dataset_info
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='isic2020',
                        help='dataset name')
    parser.add_argument('--datadir', type=str, default='',
                        help='path to dataset directory')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--log_name', type=str, default='',
                        help='experiment log name')
    parser.add_argument('--mode', type=str, default='UpperBound',
                        help='UpperBound or Fed')
    parser.add_argument('--approach', type=str, default='',
                        help='approach name')
    parser.add_argument('--model', type=str, default='ResNet18',
                        help='neural network used in training')

    return parser.parse_args()

def main():

    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')
    args.device = device

    args.n_classes, args.channel, args.im_size, \
    args.mean, args.std = get_dataset_info(args.dataset)

    _, _, _, test_ds, _ = get_dataloader(args)
    test_dl = DataLoader(dataset=test_ds, batch_size=64, shuffle=False,
                        num_workers=8, pin_memory=True, prefetch_factor=2*64)

    w_log_name = 'global_' + args.model + '_round0_' + args.log_name + '.pth'

    global_net = get_network(args)
    if device != 'cpu':
        global_net = nn.DataParallel(global_net)
        global_net.to(device)
    global_net.load_state_dict(torch.load(os.path.join('./models', args.mode, args.approach, w_log_name), map_location=device))

    y_true, y_pred = [], []

    global_net.eval()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_dl):
            x = x.to(device, non_blocking=True)
            target = target.long().to(device, non_blocking=True)

            out = global_net(x)[0]
            _, pred_label = torch.max(out.data, 1)

            y_true += target.cpu().tolist()
            y_pred += pred_label.cpu().tolist()

    print(classification_report(np.array(y_true), np.array(y_pred), digits=5))

if __name__ == '__main__':
    main()