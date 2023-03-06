import numpy as np
import torch.nn

import torchvision as tv
from torch import nn
from torch.nn import init


class ResNet34_Pretrained(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = tv.models.resnet34(
            weights=tv.models.ResNet34_Weights.DEFAULT)

#        for param in self.model.parameters():
#            param.requires_grad = False

        self.model.fc = torch.nn.Linear(512, 128)
        init.xavier_uniform_(self.model.fc.weight)

        self.drop = nn.Dropout(p=0.5, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc = torch.nn.Linear(128, 2)
        init.xavier_uniform_(self.model.fc.weight)
        init.xavier_uniform_(self.fc.weight)


        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        #x = nn.functional.interpolate(x, (224, 224), mode='bilinear')
        #x = x - torch.min(x)
        #x = x / torch.max(x)
        #x[:, 0, :, :] = (x[:, 0, :, :] - self.mean[0])/self.std[0]
        #x[:, 1, :, :] = (x[:, 1, :, :] - self.mean[1])/self.std[1]
        #x[:, 2, :, :] = (x[:, 2, :, :] - self.mean[2])/self.std[2]

        x = self.model(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
