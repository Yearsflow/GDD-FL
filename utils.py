import random
import os
import numpy as np
import torch
from dataset_utils import MNIST_truncated, CIFAR10_truncated, CIFAR100_truncated, DatasetSplit
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from networks import ResNet18, ConvNet
import torch.nn as nn
from scipy.ndimage.interpolation import rotate as scipyrotate
import torch.nn.functional as F

# fixing every seed
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def get_dataset_info(dataset):
        
    if dataset == 'mnist':
        return 10, 1, (28, 28), [0.1307], [0.3081]
    elif dataset == 'cifar10':
        return 10, 3, (32, 32), [x / 255.0 for x in [125.3, 123.0, 113.9]], [x / 255.0 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'isic2020':
        return 2, 3, (224, 224), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif dataset == 'EyePACS':
        return 5, 3, (224, 224), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise NotImplementedError('Dataset Not Supported')

def get_dataloader(args, request='dataloader'):

    num_per_class = None
    
    if args.dataset == 'mnist':

        n_train, n_val, n_public = 50000, 5000, 5000
        idxs = [_ for _ in range(60000)]
        random.shuffle(idxs)
        train_idxs = idxs[: n_train]
        val_idxs = idxs[n_train: n_train + n_val]
        public_idxs = idxs[n_train + n_val: n_train + n_val + n_public]

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])

        if args.approach in {'feddc', 'feddsa', 'feddm', 'fedgan', 'feddream'}:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081])
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])

        train_ds = MNIST_truncated(args.datadir, dataidxs=train_idxs, train=True, transform=transform_train, download=True)
        val_ds = MNIST_truncated(args.datadir, dataidxs=val_idxs, train=True, transform=transform_test, download=True)
        test_ds = MNIST_truncated(args.datadir, train=False, transform=transform_test, download=True)

        train_dl = DataLoader(dataset=train_ds, batch_size=args.train_bs, drop_last=False, shuffle=True, num_workers=16, pin_memory=True)
        val_dl = DataLoader(dataset=val_ds, batch_size=args.test_bs, shuffle=False, num_workers=16, pin_memory=True)
        test_dl = DataLoader(dataset=test_ds, batch_size=args.test_bs, shuffle=False, num_workers=16, pin_memory=True)
        
    elif args.dataset == 'cifar10':

        n_train, n_val, n_public = 40000, 5000, 5000
        idxs = [_ for _ in range(50000)]
        random.shuffle(idxs)
        train_idxs = idxs[: n_train]
        val_idxs = idxs[n_train: n_train + n_val]
        public_idxs = idxs[n_train + n_val: n_train + n_val + n_public]

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], 
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        ])

        if args.approach in {'feddc', 'feddsa', 'feddm', 'fedgan'}:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], 
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], 
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        ])

        train_ds = CIFAR10_truncated(args.datadir, dataidxs=train_idxs, train=True, transform=transform_train, download=True)
        val_ds = CIFAR10_truncated(args.datadir, dataidxs=val_idxs, train=True, transform=transform_test, download=True)
        test_ds = CIFAR10_truncated(args.datadir, train=False, transform=transform_test, download=True)

        train_dl = DataLoader(dataset=train_ds, batch_size=args.train_bs, drop_last=False, shuffle=True, num_workers=16, pin_memory=True)
        val_dl = DataLoader(dataset=val_ds, batch_size=args.test_bs, shuffle=False, num_workers=16, pin_memory=True)
        test_dl = DataLoader(dataset=test_ds, batch_size=args.test_bs, shuffle=False, num_workers=16, pin_memory=True)
        
    elif args.dataset == 'cifar100':

        n_train, n_val, n_public = 40000, 5000, 5000
        idxs = [_ for _ in range(50000)]
        random.shuffle(idxs)
        train_idxs = idxs[: n_train]
        val_idxs = idxs[n_train: n_train + n_val]
        public_idxs = idxs[n_train + n_val: n_train + n_val + n_public]

        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])

        train_ds = CIFAR100_truncated(args.datadir, dataidxs=train_idxs, train=True, transform=transform_train, download=True)
        val_ds = CIFAR100_truncated(args.datadir, dataidxs=val_idxs, train=True, transform=transform_test, download=True)
        test_ds = CIFAR100_truncated(args.datadir, train=False, transform=transform_test, download=True)

        train_dl = DataLoader(dataset=train_ds, batch_size=args.train_bs, drop_last=False, shuffle=True, num_workers=16, pin_memory=True)
        val_dl = DataLoader(dataset=val_ds, batch_size=args.test_bs, shuffle=False, num_workers=16, pin_memory=True)
        test_dl = DataLoader(dataset=test_ds, batch_size=args.test_bs, shuffle=False, num_workers=16, pin_memory=True)
        
    elif args.dataset == 'isic2020':

        n_train = [19381, 351]
        n_val, n_public = [3326, 58], [3326, 58]
        n_test = [6509, 117]
        idxs = [[_ for _ in range(32542)], [__+32542 for __ in range(584)]]
        train_idxs, val_idxs = [None for _ in range(2)], [None for _ in range(2)]
        public_idxs, test_idxs = [None for _ in range(2)], [None for _ in range(2)]
        for i in range(2):
            random.shuffle(idxs[i])
            train_idxs[i] = idxs[i][: n_train[i]]
            val_idxs[i] = idxs[i][n_train[i]: n_train[i] + n_val[i]]
            public_idxs[i] = idxs[i][n_train[i] + n_val[i]: n_train[i] + n_val[i] + n_public[i]]
            test_idxs[i] = idxs[i][n_train[i] + n_val[i] + n_public[i]: ]

        transform_train = transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.CenterCrop(224),
            transforms.RandomApply([transforms.RandomAffine(degrees=30)], p=0.3),
            transforms.RandomApply([transforms.ColorJitter(contrast=random.choice((0.5, 1.5)))], p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 5))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        total_train_idxs = train_idxs[0] + train_idxs[1]
        train_targets = [0 for _ in range(len(train_idxs[0]))] + [1 for _ in range(len(train_idxs[1]))]
        total_val_idxs = val_idxs[0] + val_idxs[1]
        val_targets = [0 for _ in range(len(val_idxs[0]))] + [1 for _ in range(len(val_idxs[1]))]
        total_test_idxs = test_idxs[0] + test_idxs[1]

        dst = datasets.ImageFolder(root=args.datadir)

        train_ds = DatasetSplit(dataset=dst, idxs=total_train_idxs, transform=transform_train, targets=train_targets)
        val_ds = DatasetSplit(dataset=dst, idxs=total_val_idxs, transform=transform_test, targets=val_targets)
        test_ds = DatasetSplit(dataset=dst, idxs=total_test_idxs, transform=transform_test)

        train_dl = DataLoader(dataset=train_ds, batch_size=args.train_bs, drop_last=False, shuffle=True, num_workers=16, pin_memory=True)
        val_dl = DataLoader(dataset=val_ds, batch_size=args.test_bs, shuffle=False, num_workers=16, pin_memory=True)
        test_dl = DataLoader(dataset=test_ds, batch_size=args.test_bs, shuffle=False, num_workers=16, pin_memory=True)

        num_per_class = {
            'train': n_train,
            'val': n_val,
            'public': n_public,
            'test': n_test
        }

        if request == 'dataloader':
            return train_dl, val_dl, test_dl, num_per_class
        elif request == 'dataset':
            return train_ds, val_ds, test_ds, num_per_class

    elif args.dataset == 'EyePACS':

        n_train = [23229, 2199, 4763, 786, 638]
        n_public, n_val = [2581, 244, 529, 87, 70], [8130, 720, 1579, 237, 240]
        n_test = [31403, 3042, 6282, 977, 966]
        idxs = [[_ for _ in range(n_train[0] + n_public[0])], 
                [_ + n_train[0] + n_public[0] for _ in range(n_train[1] + n_public[1])], 
                [_ + sum(n_train[:2]) + sum(n_public[:2]) for _ in range(n_train[2] + n_public[2])], 
                [_ + sum(n_train[:3]) + sum(n_public[:3]) for _ in range(n_train[3] + n_public[3])],
                [_ + sum(n_train[:4]) + sum(n_public[:4]) for _ in range(n_train[4] + n_public[4])]]
        train_idxs = [None for _ in range(5)]
        public_idxs = [None for _ in range(5)]
        for i in range(5):
            random.shuffle(idxs[i])
            train_idxs[i] = idxs[i][: n_train[i]]
            public_idxs[i] = idxs[i][n_train[i]:]

        transform_train = transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.CenterCrop(224),
            transforms.RandomApply([transforms.RandomAffine(degrees=30)], p=0.3),
            transforms.RandomApply([transforms.ColorJitter(contrast=random.choice((0.5, 1.5)))], p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 5))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        total_train_idxs, train_targets = [], []
        total_public_idxs, public_targets = [], []
        for c in range(5):
            total_train_idxs += train_idxs[c]
            train_targets += [c for _ in range(len(train_idxs[c]))]
            total_public_idxs += public_idxs[c]
            public_targets += [c for _ in range(len(public_idxs[c]))]

        train_dst = datasets.ImageFolder(root=os.path.join(args.datadir, 'train'))
        val_ds = datasets.ImageFolder(root=os.path.join(args.datadir, 'val'), transform=transform_test)
        test_ds = datasets.ImageFolder(root=os.path.join(args.datadir, 'test'), transform=transform_test)

        train_ds = DatasetSplit(dataset=train_dst, idxs=total_train_idxs, transform=transform_train, targets=train_targets)

        train_dl = DataLoader(dataset=train_ds, batch_size=args.train_bs, drop_last=False, shuffle=True, num_workers=8, prefetch_factor=16*64, pin_memory=True)
        val_dl = DataLoader(dataset=val_ds, batch_size=args.test_bs, shuffle=False, num_workers=8, prefetch_factor=16*64, pin_memory=True)
        test_dl = DataLoader(dataset=test_ds, batch_size=args.test_bs, shuffle=False, num_workers=8, prefetch_factor=16*64, pin_memory=True)

        num_per_class = {
            'train': n_train,
            'val': n_val,
            'public': n_public,
            'test': n_test
        }

    if request == 'dataloader':
        return train_dl, val_dl, test_dl, num_per_class
    elif request == 'dataset':
        return train_ds, val_ds, test_ds, num_per_class
    
def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']: # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling
    
def get_network(args):

    if args.dataset in ['mnist', 'cifar10']:
        n_classes = 10
        im_size = (32, 32)
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'isic2020':
        n_classes = 2
        im_size = (224, 224)
    elif args.dataset == 'EyePACS':
        n_classes = 5
        im_size = (224, 224)
    else:
        raise NotImplementedError('Dataset Not Supported')
    
    if args.dataset == 'mnist':
        n_channel = 1
    else:
        n_channel = 3
    
    if args.model == 'ResNet18':
        net = ResNet18(args=args, channel=n_channel, num_classes=n_classes)
    elif args.model == 'ConvNet':
        net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
        net = ConvNet(channel=n_channel, num_classes=n_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    if args.dataset in {'isic2020', 'EyePACS'}:
        num_ftrs = net.classifier.in_features
        net.classifier = nn.Linear(num_ftrs * 7 * 7, n_classes)
    
    return net

def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images

class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5

def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1

def DiffAugment(x, strategy='', seed=-1, param=None):
    if strategy == 'None' or strategy == 'none' or strategy == '':
        return x
    
    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True
    
    param.latestseed = seed

    if strategy:
        if param.aug_mode == 'M': #original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s' % param.aug_mode)
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        tmp = theta[0].clone()
        theta[:] = tmp
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        tmp = theta[0].clone()
        theta[:] = tmp
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese: # Siamese augmentation:
        tmp = randf[0].clone()
        randf[:] = tmp
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        tmp = randb[0].clone()
        randb[:] = tmp
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        tmp = rands[0].clone()
        rands[:] = tmp
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        tmp = randc[0].clone()
        randc[:] = tmp
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        tmp = translation_x[0].clone()
        translation_x[:] = tmp
        tmp = translation_y[0].clone()
        translation_y[:] = tmp
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        tmp = offset_x[0].clone()
        offset_x[:] = tmp
        tmp = offset_y[0].clone()
        offset_y[:] = tmp
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}

def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 3, 5
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc' % ipc)
    return outer_loop, inner_loop

def match_loss(gw_syn, gw_real, args, appr_args):
    dis = torch.tensor(0.0).to(args.device)

    if appr_args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif appr_args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif appr_args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s' % appr_args.dis_metric)

    return dis

def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis