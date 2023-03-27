import numpy as np
import torch.nn

from cli_program.settings.behaviour_settings import BASE_MODEL_PATH
from model.NNModels.ResNet34_pre import ResNet34SoftMax


class ResNet34_4to2(ResNet34SoftMax):

    def __init__(self):
        super().__init__(4)

        state = torch.load(BASE_MODEL_PATH)
        self.load_state_dict(state)


    def forward(self, x):

        x = super().forward(x)
        x = x > 0.5
        x = torch.stack((x[:,1] | x[:,3], x[:,2] | x[:,3])).transpose(0,1).float()

        return x
