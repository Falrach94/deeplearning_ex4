import numpy as np
import torch.nn

import torchvision as tv
from torch import nn
from torch.nn import init

from cli_program.settings.behaviour_settings import BASE_MODEL_PATH


class ResNet50v2_Pretrained(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = tv.models.wide_resnet50_2(
            weights=tv.models.Wide_ResNet50_2_Weights.DEFAULT)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 4),
            nn.Softmax(1)
        )
        for module in self.model.fc.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)

       # state = torch.load(BASE_MODEL_PATH)
       # self.load_state_dict(state)

#        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return x
