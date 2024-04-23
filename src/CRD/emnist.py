from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
    else:
        data_folder = './data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class EMNISTInstance(datasets.EMNIST):
    """CIFAR100Instance Dataset.
    """
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


def get_emnist_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    """
    emnist
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    if is_instance:
        train_set = EMNISTInstance(root=data_folder,
                                   split='letters',
                                     download=True,
                                     train=True,
                                     transform=train_transform)
        n_data = len(train_set)

    else:
        train_set = datasets.EMNIST(root=data_folder,
                                    split='letters',
                                      download=True,
                                      train=True,
                                      transform=train_transform)
        for i in range(len(train_set.targets)):
            train_set.targets[i] = train_set.targets[i] - 1

        print(train_set.targets[:1000])
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.EMNIST(root=data_folder,
                               split='letters',
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    for i in range(len(test_set.targets)):
        test_set.targets[i] = test_set.targets[i] - 1
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader




class EMNISTInstanceSample_all(datasets.EMNIST):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, root,split='letters',train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root,split=split,train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample
        self.data = self.data[:10000].numpy()
        for i in range(len(self.targets)):
            self.targets[i] = self.targets[i] - 1
        self.targets = self.targets[:10000].numpy()
        num_classes = 26
        if self.train:
            num_samples = len(self.data)
            label = self.targets
        else:
            num_samples = len(self.data)
            label = self.targets
        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)


    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx

def get_emnist_dataloaders_sample(batch_size=128, num_workers=4, k=4096, mode='exact',
                                    is_sample=True, percent=1.0):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    train_set = EMNISTInstanceSample_all(root=data_folder,
                                         split='letters',
                                       download=True,
                                       train=True,
                                       transform=train_transform,
                                       k=k,
                                       mode=mode,
                                       is_sample=is_sample,
                                       percent=percent)
    n_data = len(train_set)
    print(n_data)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.EMNIST(root=data_folder,
                                     split='letters',
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    for i in range(len(test_set.targets)):
        test_set.targets[i] = test_set.targets[i] - 1
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader, n_data
