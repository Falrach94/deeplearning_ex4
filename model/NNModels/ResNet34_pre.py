import cv2
import numpy as np
import torch.nn
import torchvision

import torchvision as tv
from torch import nn
from torch.nn import init, Sequential
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
    def __init__(self, out_cnt, pre_path=None, multi_layer=True):
        super().__init__(load_weights=(pre_path is None))

        if multi_layer:
            self.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(512, 1024),
                nn.Dropout(p=0.5),
                nn.ReLU(inplace=True),
                nn.Linear(1024, out_cnt),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(512, out_cnt),
                nn.Sigmoid()
            )

        if pre_path is not None:
            state = torch.load(pre_path)
            self.load_state_dict(state)
        else:
            for module in self.fc.modules():
                if isinstance(module, nn.Linear):
                    init.xavier_uniform_(module.weight)


class ResNet34SigAux(ResNet34Sig):
    def __init__(self, out_cnt, pre_path=None, multi_layer=True):

        super().__init__(out_cnt, pre_path, multi_layer)

        self.aux = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 4),
            nn.Softmax(dim=1)
        )

        if pre_path is None:
            for module in self.aux.modules():
                if isinstance(module, nn.Linear):
                    init.xavier_uniform_(module.weight)

        self.aux_prediction = None

        self.ks = 5
        self.padding = self.ks // 2
        var = 1
        self.gauss = torch.tensor(cv2.getGaussianKernel(self.ks, var),
                                  dtype=torch.float)[:, 0].cuda()

    def cuda(self):
        self.gauss = self.gauss.cuda()
        return super().cuda()

    def cpu(self):
        self.gauss = self.gauss.cpu()
        return super().cpu()

    def forward(self, x):

        x = x[:, :1, :, :]

        x = torch.nn.functional.conv2d(x, weight=self.gauss.view(1, 1, -1, 1), padding=(self.padding, 0))
        x = torch.nn.functional.conv2d(x, weight=self.gauss.view(1, 1, 1, -1), padding=(0, self.padding))

        x = x.repeat(1, 3, 1, 1)



      #  x = ((x - x.min())/(x.max()-x.min())-0.5)*2

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


class ResNet34SoftMax(ResNet34Base):
    def __init__(self, out_cnt, pre_path=None):
        super().__init__(load_weights=(pre_path is None))

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 1024),
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



class ResNet34Combined(nn.Module):

    def __init__(self, distinction_path, defect_path):
        super().__init__()

        self.dist = ResNet34Sig(1, distinction_path, multi_layer=True)
        self.defect = ResNet34Sig(2, defect_path)

        #self.dist.requires_grad_(False)
        #self.defect.requires_grad_(False)

    def forward(self, x):
        is_defect = self.dist(x)
        is_defect = is_defect.repeat(1, 2)
        x = self.defect(x)
        x[is_defect < 0.5] = 0

        return x
