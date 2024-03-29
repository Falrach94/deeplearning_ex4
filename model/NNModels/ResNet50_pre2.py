import numpy as np
import torch.nn

import torchvision as tv
from torch import nn
from torch.nn import init
from torchvision.models import ResNet
from torchvision.models.quantization.resnet import QuantizableBottleneck
from torchvision.models.resnet import BasicBlock

from cli_program.settings.behaviour_settings import BASE_MODEL_PATH


class ResNet50Base(ResNet):
    def __init__(self, load_weights=True):
        super().__init__(QuantizableBottleneck, [3, 4, 6, 3])

        if load_weights:
            weights = tv.models.ResNet50_Weights.DEFAULT
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

class ResNet50Sig(ResNet50Base):
    def __init__(self, out_cnt, pre_path=None, multi_layer=True):
        super().__init__(load_weights=(pre_path is None))

        if multi_layer:
            self.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(2048, 1028),
                nn.Dropout(p=0.5),
                nn.ReLU(inplace=True),
                nn.Linear(1028, out_cnt),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(2048, out_cnt),
                nn.Sigmoid()
            )


        if pre_path is not None:
            state = torch.load(pre_path)
            self.load_state_dict(state)
        else:
            for module in self.fc.modules():
                if isinstance(module, nn.Linear):
                    init.xavier_uniform_(module.weight)


class ResNet50SigAux(ResNet50Sig):
    def __init__(self, out_cnt, pre_path=None, multi_layer=True):

        super().__init__(out_cnt, pre_path, multi_layer)

        self.aux = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 4),
            nn.Softmax(dim=1)
        )

        if pre_path is None:
            for module in self.aux.modules():
                if isinstance(module, nn.Linear):
                    init.xavier_uniform_(module.weight)

        self.aux_prediction = None

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

        self.aux_prediction = self.aux(x)

        x = self.fc(x)
        return x

class ResNet50SoftMax(ResNet50Base):
    def __init__(self, out_cnt, pre_path=None):
        super().__init__(load_weights=(pre_path is None))

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_cnt),
            nn.Softmax(dim=1)
        )

        if pre_path is not None:
            state = torch.load(pre_path)
            self.load_state_dict(state)
        else:
            for module in self.fc.modules():
                if isinstance(module, nn.Linear):
                    init.xavier_uniform_(module.weight)

