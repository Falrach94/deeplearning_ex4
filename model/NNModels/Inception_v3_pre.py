import numpy as np
import torch.nn

import torchvision as tv

class InceptionV3_Pretrained(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = tv.models.inception_v3(
            weights=tv.models.Inception_V3_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = torch.nn.Linear(2048, 2)

        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x
