import numpy as np
import torch.nn

import torchvision as tv
from torch import nn
from torch.nn import init


class ResNet34_Pretrained2(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = tv.models.resnet34(
            weights=tv.models.ResNet34_Weights.DEFAULT)

#        for param in self.model.parameters():
#            param.requires_grad = False

        self.fc2 = torch.nn.Linear(128, 2)

        self.model.fc = torch.nn.Linear(512, 128)
        init.xavier_uniform_(self.model.fc.weight)
        init.xavier_uniform_(self.fc2.weight)

        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.5)
        self.feature_learning = True

        #self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        #self.std = torch.tensor([0.229, 0.224, 0.225]).cuda()
    def forward(self, x):
        x = self.model(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
