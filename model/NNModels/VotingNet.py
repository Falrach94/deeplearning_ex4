import os

import numpy as np
import torch
import torch.nn as nn

from model.NNModels.ResNet import ResNet
from model.NNModels.ResNet34_pre import ResNet34_Pretrained
from model.preprocessing.freq_filter import FrequencyFilter

VOTER_CNT = 5
class VotingNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.voters = [ResNet34_Pretrained() for _ in range(VOTER_CNT)]

        self.voter1 = self.voters[0]
        self.voter2 = self.voters[1]
        self.voter3 = self.voters[2]
        self.voter4 = self.voters[3]
        self.voter5 = self.voters[4]

        self.sig = nn.Sigmoid()

    @staticmethod
    def create():
        net = VotingNet()

        state = [torch.load(f'assets/best_model{i}.ckp') for i in range(VOTER_CNT)]
        for n, s in zip(net.voters, state):
            n.load_state_dict(s)


        return net

    def forward(self, x):
        y = [net(x)[:, None, :] for net in self.voters]
        x = torch.cat(y, dim=1)
        conf = torch.maximum(x, 1-x)
        max_conf, ix = torch.max(conf, dim=1, keepdim=True)

        x = x[conf == max_conf].view(-1, 2)

        return x
