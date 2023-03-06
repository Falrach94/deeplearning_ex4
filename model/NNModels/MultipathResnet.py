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

        path = nn.Sequential(
            copy.deepcopy(self.layer3),
            copy.deepcopy(self.layer4),
            copy.deepcopy(self.avgpool),
            nn.Flatten(),
            nn.Linear(512, self.inter_cnt),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
        )

        self.layer3 = None
        self.layer4 = None
        self.avgpool = None

        self.init_stage = [
            self.conv1,
            self.bn1,
            self.maxpool,
            self.layer1,
            self.layer2
        ]

        self.extraction_paths = [copy.deepcopy(path) for _ in range(path_cnt)]
        self.fc_single = [nn.Linear(self.inter_cnt, 2) for _ in range(path_cnt)]

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

        self.fc = nn.Linear(self.inter_cnt*path_cnt, 2)

        state = torch.load(self.base_path)
        self.load_state_dict(state)

        self.sig = nn.Sigmoid()

        self.use_path = None

    def set_path(self, path, train):
        for param in self.parameters():
            param.requires_grad = False

        if path is None:
            if train:
                init.xavier_uniform_(self.fc.weight)

            for param in self.fc.parameters():
                param.requires_grad = True

        else:
            for param in self.extraction_paths[path].parameters():
                param.requires_grad = True

            if train:
                init.xavier_uniform_(self.fc_single[path].weight)
                for m in self.extraction_paths[path]():
                    if isinstance(m, nn.Linear):
                        init.xavier_uniform_(m.weight)

            for param in self.fc_single[path].parameters():
                param.requires_grad = True

        self.use_path = path

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        if self.use_path is None:
            y = [path(x)[:, :, None] for path in self.extraction_paths]
            x = torch.concat(y, dim=2)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = self.extraction_paths[self.use_path](x)
            x = self.fc_single[self.use_path](x)

        x = self.sig(x)

        return x
