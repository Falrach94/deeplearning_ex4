import copy

import numpy as np
import torch.nn

import torchvision as tv
from torch import nn
from torch.nn import init
from torchvision.models.resnet import BasicBlock, ResNet


class MultipathResNet34(ResNet):

    base_path = 'assets/base_model.ckp'
    inter_cnt = 32

    def __init__(self, path_cnt):
        super().__init__(BasicBlock, [3, 4, 6, 3])

        weights = tv.models.ResNet34_Weights.DEFAULT
        self.load_state_dict(weights.get_state_dict(True))


        self.init_stage = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2
        )

        path = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
            nn.Flatten(),
            nn.Linear(512, self.inter_cnt),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
        )

        self.extraction_paths = [copy.deepcopy(path) for _ in range(path_cnt)]
        for path in self.extraction_paths:
            for module in path.modules():
                if isinstance(module, nn.Linear):
                    init.xavier_uniform_(module.weight)

        self.fc_single = [nn.Linear(self.inter_cnt, 2) for _ in range(path_cnt)]
        for fc in self.fc_single:
            init.xavier_uniform_(fc.weight)

        self.fc = nn.Linear(self.inter_cnt*path_cnt, 2)
        init.xavier_uniform_(self.fc.weight)

        self.conv1 = None
        self.bn1 = None
        self.maxpool = None
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.avgpool = None

        self.path1 = self.extraction_paths[0]
        self.path2 = self.extraction_paths[1]
        self.path3 = self.extraction_paths[2]
        self.path4 = self.extraction_paths[3]
        self.path5 = self.extraction_paths[4]

        self.fc_single1 = self.fc_single[0]
        self.fc_single2 = self.fc_single[1]
        self.fc_single3 = self.fc_single[2]
        self.fc_single4 = self.fc_single[3]
        self.fc_single5 = self.fc_single[4]

        self.train_ll = False

        self.sig = nn.Sigmoid()

        self.active_path = None

    def train(self, mode=True):
        super().train(mode)

        if not self.train_ll:
            self.init_stage.eval()

        if mode:
            for param in self.parameters():
                param.grad = None

    def set_path(self, path, train):
        self.active_path = path
        self.requires_grad_(False)

        if not train:
            return

        if path == 0:
            self.train_ll = True
        else:
            self.train_ll = False

        if path is None:
            self.fc.requires_grad_(True)
        else:
            if self.train_ll:
                self.init_stage.requires_grad_(True)
            self.extraction_paths[path].requires_grad_(True)
            self.fc_single[path].requires_grad_(True)

    def forward(self, x):

        x = self.init_stage(x)

        if self.active_path is None:
            y = [path(x)[:, :, None] for path in self.extraction_paths]
            x = torch.concat(y, dim=2)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = self.extraction_paths[self.active_path](x)
            x = self.fc_single[self.active_path](x)

        x = self.sig(x)

        return x
