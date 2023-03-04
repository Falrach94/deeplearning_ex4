import torch
from torch import nn
import torchvision as tv
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock


class SkipEncoder(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [3, 4, 6, 3])
        weights = tv.models.ResNet34_Weights.DEFAULT
        self.load_state_dict(weights.get_state_dict(True))

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip1 = x  # 300x300 -> 150x150
        x = self.maxpool(x)

        skip2 = x  # -> 75x75
        x = self.layer1(x)
        skip3 = x  # -> 75x75
        x = self.layer2(x)
        skip4 = x  # -> 38x38
        x = self.layer3(x)
        skip5 = x  # -> 19x19
        x = self.layer4(x)
        # -> 512x10x10

        return x, (skip1, skip2, skip3, skip4, skip5)
