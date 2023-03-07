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
            self.fc
        )

        self.path1 = copy.deepcopy(path)
        self.path2 = copy.deepcopy(path)

        self.paths = [self.path1, self.path2]

        self.sel = 0

    def select(self, i):
        self.sel = i
        self.requires_grad_(False)
        self.paths[i].requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)

        self.init.eval()

    def forward(self, x):
        x = self.init(x)
        x = self.paths[self.sel](x)

        return x
