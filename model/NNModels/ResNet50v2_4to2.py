import numpy as np
import torch.nn

import torchvision as tv
from torch import nn
from torch.nn import init


class ResNet50v2_4to2(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = tv.models.wide_resnet50_2(
            weights=tv.models.Wide_ResNet50_2_Weights.DEFAULT)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 4),
            nn.Softmax(1)
        )
#        for module in self.model.fc.modules():
#            if isinstance(module, nn.Linear):
#                init.xavier_uniform_(module.weight)

#        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)

        x = x > 0.5
        x = torch.stack((x[:,1] | x[:,3], x[:,2] | x[:,3])).transpose(0,1).float()
        
        return x
