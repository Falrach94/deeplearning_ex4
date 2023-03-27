import numpy as np
import torch.nn

import torchvision as tv
from torch import nn
from torch.nn import init
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock

from cli_program.settings.behaviour_settings import BASE_MODEL_PATH

class ResNet34Base(ResNet):
    def __init__(self, load_weights=True):
        super().__init__(BasicBlock, [3, 4, 6, 3])

        if load_weights:
            weights = tv.models.ResNet34_Weights.DEFAULT
            self.load_state_dict(weights.get_state_dict(True))

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
        return x


class ResNet34Sig(ResNet34Base):
    def __init__(self, out_cnt, pre_path=None):
        super().__init__(load_weights=(pre_path is None))

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_cnt),
            nn.Sigmoid()
        )

        if pre_path is not None:
            state = torch.load(pre_path)
            self.load_state_dict(state)
        else:
            for module in self.fc.modules():
                if isinstance(module, nn.Linear):
                    init.xavier_uniform_(module.weight)


class ResNet34SoftMax(ResNet34Base):
    def __init__(self, out_cnt, pre_path=None):
        super().__init__(load_weights=(pre_path is None))

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_cnt),
            nn.Softmax(dim=1)
        )

        if pre_path is not None:
            state = torch.load(pre_path)
            self.load_state_dict(state)
        else:
            for module in self.fc.modules():
                if isinstance(module, nn.Linear):
                    init.xavier_uniform_(module.weight)


class ResNet34Combined(nn.Module):

    def __init__(self, distinction_path, defect_path):
        super().__init__()

        self.dist = ResNet34Sig(1, distinction_path)
        self.defect = ResNet34Sig(2, defect_path)

        self.dist.requires_grad_(False)
        self.defect.requires_grad_(False)

    def forward(self, x):
        is_defect = self.dist(x)
        is_defect = is_defect.repeat(1, 2)
        x = self.defect(x)
        x[is_defect < 0.5] = 0

        return x
