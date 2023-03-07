import copy

import numpy as np
import torch.nn

import torchvision as tv
from torch import nn
from torch.nn import init
from torchvision.models.resnet import BasicBlock, ResNet


class TestResNet34(ResNet):

    def __init__(self):
        super().__init__(BasicBlock, [3, 4, 6, 3])

        weights = tv.models.ResNet34_Weights.DEFAULT
        self.load_state_dict(weights.get_state_dict(True))

        self.requires_grad_(False)

        self.init = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2
        )

        path = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
            nn.Flatten(),
            nn.Linear(512, 2)
        )

        self.path1 = copy.deepcopy(path)
        self.path2 = copy.deepcopy(path)
        self.path3 = copy.deepcopy(path)
        self.path4 = copy.deepcopy(path)
        self.path5 = copy.deepcopy(path)

        self.current_path = None
        self.paths = [self.path1, self.path2, self.path3, self.path4, self.path5]

        self.sel = 0

    def set_path(self, path, train):
        self.sel = path
        self.current_path = self.paths[path]
        self.requires_grad_(False)

        self.current_path.requires_grad_(train)
        self.init.requires_grad_(path == 0 and train)

    def train(self, mode=True):
        super().train(mode)

        if self.sel != 0:
            self.init.eval()

        if mode:
            for param in self.parameters():
                param.grad = None


    def forward(self, x):
        x = self.init(x)
        x = self.current_path(x)

        return x
