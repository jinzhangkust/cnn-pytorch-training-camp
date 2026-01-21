"""
Author: Dr. Jin Zhang
E-mail: j.zhang@kust.edu.cn
URL: https://github.com/jinzhangkust
Dept: Kunming University of Science and Technology (KUST)
Created on 2025.07.05
Modified on 2026.01.04
"""

import torch
import torchvision
from torch.utils.data import Dataset

import os
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageFilter


normalize = torchvision.transforms.Normalize(mean=[0.5561, 0.5706, 0.5491], std=[0.1833, 0.1916, 0.2061])


class GaussianBlur:
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class RandAugment:
    def __init__(self, k):
        self.k = k
        self.augment_pool = [torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2),
            torchvision.transforms.RandomApply([GaussianBlur([0.1, 1.5])], p=0.7),
            torchvision.transforms.RandomHorizontalFlip()]

    def __call__(self, im):
        ops = random.choices(self.augment_pool, k=self.k)
        for op in ops:
            if random.random() < 0.5:
                im = op(im)
        return im



class TransformOnce:
    def __init__(self, imsize):
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(imsize, scale=((imsize - 10) / 400, (imsize + 10) / 400)),
            # torchvision.transforms.RandomCrop(imsize),
            RandAugment(k=3),
            torchvision.transforms.ToTensor(),
            normalize])
    def __call__(self, x):
        return self.transform(x)


class TransformTwice:
    def __init__(self, imsize):
        self.transform_weak = torchvision.transforms.Compose([
            #torchvision.transforms.CenterCrop(imsize),  # for test
            torchvision.transforms.RandomResizedCrop(imsize, scale=((imsize-10)/400, (imsize+10)/400)),  # for train
            torchvision.transforms.ToTensor(),
            normalize])
        self.transform_strong = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(imsize, scale=((imsize-10)/400, (imsize+10)/400)),
            torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2),
            torchvision.transforms.RandomApply([GaussianBlur([0.1, 1.5])], p=0.7),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize])
    def __call__(self, x):
        return [self.transform_weak(x), self.transform_strong(x)]


class stibium_data(Dataset):
    def __init__(self, imsize=300):
        root = '/home/libsv/Data/StibiumData/normal'
        self.imfiles = []
        self.labels = []
        self.imsize = imsize
        for class_folder in os.listdir(root):  # 1 2 3 4 5
            file_path = root + '/' + class_folder
            clip_folder = os.listdir(file_path)  # c1_*  c2_*  c3_*  c4_*  c5_*
            for item in clip_folder:
                full_path = file_path + '/' +item
                #print(full_path)   # /home/libsv/Data/StibiumData/normal/c1_1/
                file = os.path.join(full_path, '*_1.jpg')  # /home/libsv/Data/StibiumData/normal/c1_1/stibium_c1_1.jpg
                for im in glob.glob(file):
                    self.imfiles.append(im)
                    label = int(class_folder[0]) - 1
                    #print(f"label: {label}")
                    self.labels.append(label)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(self.imsize, scale=((self.imsize-10) / 400, (self.imsize+10) / 400)),
            #torchvision.transforms.RandomCrop(self.imsize),
            RandAugment(k=3),
            torchvision.transforms.ToTensor(),
            normalize])

        self.transform_twice = TransformTwice(self.imsize)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = self.imfiles[idx]
        label = self.labels[idx]
        im = Image.open(path).convert("RGB")
        im = self.transform(im)
        return idx, im, label


def get_stibium_data():
    full_data = stibium_data()
    train_size = int(0.7 * len(full_data))
    val_size = int(0.15 * len(full_data))
    test_size = len(full_data) - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(full_data, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(32))
    return train_data, val_data, test_data


def get_ssl_froth_loader(opt):
    full_data = stibium_data()
    train_size = int(0.6 * len(full_data))
    val_size = int(0.2 * len(full_data))
    test_size = len(full_data) - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(full_data, [train_size, val_size, test_size],
                                                                    generator=torch.Generator().manual_seed(42))
    random.seed(42)
    train_index = list(range(train_size))
    random.shuffle(train_index)
    train_labeled_index = train_index[:int(train_size * 0.2)]
    train_unlabeled_index = train_index[int(train_size * 0.2):]

    train_labeled_sampler = torch.utils.data.SubsetRandomSampler(train_labeled_index)
    train_unlabeled_sampler = torch.utils.data.SubsetRandomSampler(train_unlabeled_index)

    train_labeled_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, sampler=train_labeled_sampler)
    train_unlabeled_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, sampler=train_unlabeled_sampler)
    val_loader = torch.utils.data.DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    return train_labeled_loader, train_unlabeled_loader, val_loader, test_loader


def get_full_data_1():
    full_data = torchvision.datasets.ImageFolder(root='/home/libsv/Data/StibiumData/normal', transform=TransformTwice(300))


class IndexedImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        im, label = super(IndexedImageFolder, self).__getitem__(index)
        return index, im, label


def get_froth_data_2():
    transform = TransformOnce(300)
    #train_data = IndexedImageFolder(root='/home/libsv/Data/StibiumData/train', transform=transform)
    #val_data = IndexedImageFolder(root='/home/libsv/Data/StibiumData/val', transform=transform)

    full_data = IndexedImageFolder(root='/home/libsv/Data/StibiumData/normal', transform=transform)
    train_size = int(0.6 * len(full_data))
    val_size = int(0.2 * len(full_data))
    test_size = len(full_data) - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(full_data, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    return train_data, val_data, test_data
    #return train_data, val_data


