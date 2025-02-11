import torch.utils.data as data
import numpy as np
import torchvision
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
import cv2

class ImageFolder_A(DatasetFolder):
    
    def __init__(self, root: str, transform = None, target_transform = None,
                loader = default_loader, is_valid_file = None):
        super(ImageFolder_A, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None, 
                                            transform = transform, 
                                            target_transform = target_transform, 
                                            is_valid_file = is_valid_file)
        self.imgs = self.samples

    def __getitem__(self, index: int):

        path, target = self.samples[index]
        sample = cv2.imread(path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            sample = self.transform(image=sample)['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

class TensorDataset(data.Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

class DatasetSplit(data.Dataset):

    def __init__(self, dataset, idxs, transform=None, targets=None):
        self.dataset = dataset
        self.idxs = idxs
        self.transform = transform
        self.targets = targets
    
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, index):
        image, label = self.dataset[int(self.idxs[index])]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

# Based on original torchvision dataset, adding the function of subseting local client data via dataidxs.

class MNIST_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=True):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, targets = mnist_dataobj.train_data, np.array(mnist_dataobj.train_labels)
            else:
                data, targets = mnist_dataobj.test_data, np.array(mnist_dataobj.test_labels)
        else:
            data = np.array(mnist_dataobj.data)
            targets = np.array(mnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # img = Image.fromarray(img)
        # print("mnist img:", img)
        # print("mnist target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
    
class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=True):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, targets = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, targets = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            targets = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, targets = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, targets = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            targets = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # img = Image.fromarray(img)
        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)