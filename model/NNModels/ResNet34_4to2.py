import numpy as np
import torch.nn

import torchvision as tv
from torch import nn
from torch.nn import init
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock

from cli_program.settings.behaviour_settings import BASE_MODEL_PATH


class ResNet34_4to2(ResNet):

    def __init__(self):
        super().__init__(BasicBlock, [3, 4, 6, 3])

        weights = tv.models.ResNet34_Weights.DEFAULT
        self.load_state_dict(weights.get_state_dict(True))

        self.fc = nn.Sequential(
            nn.Linear(512, 4),
            nn.Softmax(dim=1)
        )
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)

        state = torch.load(BASE_MODEL_PATH)
        self.load_state_dict(state)





    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        x = x > 0.5
        x = torch.stack((x[:,1] | x[:,3], x[:,2] | x[:,3])).transpose(1,2)

      #  x = self.sigmoid(x)
        return x
