import numpy as np
import torch.nn

import torchvision as tv
from torch import nn
from torch.nn import init
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock

from cli_program.settings.behaviour_settings import BASE_MODEL_PATH


class ResNet34_Pretrained(ResNet):

    def __init__(self, label_cnt):
        super().__init__(BasicBlock, [3, 4, 6, 3])

        weights = tv.models.ResNet34_Weights.DEFAULT
        self.load_state_dict(weights.get_state_dict(True))

        #self.fc = torch.nn.Linear(512, label_cnt)
        #init.xavier_uniform_(self.fc.weight)
        #state = torch.load(BASE_MODEL_PATH)
        #self.load_state_dict(state)

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Softmax(dim=1)
        )
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)


        #        self.fc = torch.nn.Linear(512, 2)
#        init.xavier_uniform_(self.fc.weight)


     #   self.sigmoid = torch.nn.Sigmoid()

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



      #  x = self.sigmoid(x)
        return x
