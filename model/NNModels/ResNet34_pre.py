import numpy as np
import torch.nn

import torchvision as tv

class ResNet34_Pretrained(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = tv.models.resnet34(
            weights=tv.models.ResNet34_Weights.DEFAULT)

        #for param in self.model.parameters():
        #    param.requires_grad = False

        self.model.fc = torch.nn.Linear(512, 2)

        self.sigmoid = torch.nn.Sigmoid()

        self.feature_learning = True

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x
