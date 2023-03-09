import numpy as np
import torch.nn

import torchvision as tv
from torch import nn
from torch.nn import init


class ResNet50v2_Pretrained(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = tv.models.wide_resnet50_2(
            weights=tv.models.Wide_ResNet50_2_Weights.DEFAULT)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)
        )
        for module in self.model.fc.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x
