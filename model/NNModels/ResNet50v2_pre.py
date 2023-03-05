import numpy as np
import torch.nn

import torchvision as tv
from torch.nn import init


class ResNet50v2_Pretrained(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = tv.models.wide_resnet50_2(
            weights=tv.models.Wide_ResNet50_2_Weights.DEFAULT)
        self.model.fc = torch.nn.Linear(2048, 2)
        init.xavier_uniform_(self.model.fc.weight)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x
