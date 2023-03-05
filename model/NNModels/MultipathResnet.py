import copy

import numpy as np
import torch.nn

import torchvision as tv
from torch import nn
from torch.nn import init
from torchvision.models.resnet import BasicBlock, ResNet


class MultipathResNet34(ResNet):

    def __init__(self):
        super().__init__(BasicBlock, [3, 4, 6, 3])
        weights = tv.models.ResNet34_Weights.DEFAULT
        self.load_state_dict(weights.get_state_dict(True))

        self.layer3_2 = copy.deepcopy(self.layer3)
        self.layer4_2 = copy.deepcopy(self.layer4)
        self.avgpool_2 = copy.deepcopy(self.avgpool)

        self.fc = nn.Linear(512, 16)
        self.fc2 = nn.Linear(512, 16)

        self.fc3 = nn.Linear(32, 2)

        init.xavier_uniform_(self.fc.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.5)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x2 = x

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x2 = self.layer3_2(x2)
        x2 = self.layer4_2(x2)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc2(x2)

        x = torch.concat((x, x2), dim=1)
        x = self.drop(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.sig(x)

        return x
