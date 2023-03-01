import time
from random import random

import pylab as p
from torch.utils.data import Dataset
import torch
import pandas as pd
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

from model.preprocessing.dif_filter import DiffusionFilter
from model.preprocessing.freq_filter import FrequencyFilter

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):

    def __init__(self, data, mode, transform_mode, class_cnt):
        super().__init__()

        self.train = (mode == "train")
        self.time_ns = 0
        self._data = data
        self.single_label = class_cnt == 1

        if transform_mode:
            transform_mode = 2

        transforms = [tv.transforms.Compose([tv.transforms.ToPILImage(),
                                             tv.transforms.ToTensor(),
                                             tv.transforms.Normalize(train_mean, train_std)]),
                      tv.transforms.Compose([tv.transforms.ToPILImage(),
                                             tv.transforms.ToTensor(),
                                             tv.transforms.Normalize(train_mean, train_std),
                                             tv.transforms.RandomHorizontalFlip(),
                                             tv.transforms.RandomVerticalFlip(),
                                             tv.transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))]),
                      tv.transforms.Compose([tv.transforms.ToPILImage(),
                                             tv.transforms.ToTensor(),
                                             tv.transforms.Normalize(train_mean, train_std),
                                             tv.transforms.GaussianBlur(7),
                                             tv.transforms.RandomAutocontrast()])]

        if self.train:
            self._transform = transforms[transform_mode]
        else:
            self._transform = transforms[0]

        self.images = [None]*len(self._data)
        self.labels = torch.empty(size=(len(self._data), 2))

        print('loading ' + ('training' if self.train else 'eval') + ' data...')

        self._data.reset_index(inplace=True)

        for i, row in self._data.iterrows():
            self.labels[i, 0] = float(row['crack'])
            if self.single_label and self.train:
                self.labels[i, 1] = 0
            else:
                self.labels[i, 1] = float(row['inactive'])


    def __len__(self):
        return len(self._data)

    def get_forest_item(self, i):
        item = self._data[i]
        images = []
        labels = []
        for j in range(len(item)):
            rel_path = f'assets/{item[j][0]}'
            image = torch.tensor(imread(rel_path))
            image = gray2rgb(image)
            image = self._transform(image)
            #image = self.ff.filter(image)
            images.append(image[None,:])



            label = torch.tensor([float(item[j][1]),
                                  float(item[j][2])])
            label = label.reshape(2)
            labels.append(label[None,:])

        labels = torch.cat(labels)
        images = torch.cat(images)
        return images, labels

    def reset_time(self):
        self.time_ns = 0

    def get_time(self):
        return self.time_ns / 10**9

    def __getitem__(self, idx):
        # '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.images[idx] is None:
            rel_path = f'assets/{self._data.loc[idx, "filename"]}'
            image = torch.tensor(imread(rel_path))
            image = gray2rgb(image)
            self.images[idx] = image

#        image = self.images[idx][None, ...].repeat([3, 1, 1])
        image = self._transform(self.images[idx])

        return image, self.labels[idx]
        # '''
        '''
        t = time.time_ns()


        if len(self._data.shape) == 3:
            return self.get_forest_item(idx)

        rel_path = f'assets/{self._data.iloc[idx, 0]}'

        image = torch.tensor(imread(rel_path))
        image = gray2rgb(image)
        image = self._transform(image)

        label = self._data.iloc[idx, 1:]

        #if self.single_label:
        #    label = torch.tensor([float(label['crack'])])
        #    label = label.reshape(1)
        #else:
        label = torch.tensor([float(label['crack']),
                             float(label['inactive'])])
        if self.single_label and self.train:
            label[1] = 0.5
        label = label.reshape(2)

        self.time_ns += time.time_ns()-t

        return image, label
        '''