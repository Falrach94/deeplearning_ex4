import os

import numpy as np
import torch
import torch.nn as nn

from model.NNModels.ResNet import ResNet
from model.NNModels.ResNet34_pre import ResNet34_Pretrained
from model.preprocessing.freq_filter import FrequencyFilter


class VotingNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.voters = [ResNet34_Pretrained() for _ in range(5)]

        self.voter1 = self.voters[0]
        self.voter2 = self.voters[1]
        self.voter3 = self.voters[2]
        self.voter4 = self.voters[3]
        self.voter5 = self.voters[4]

        self.sig = nn.Sigmoid()

    @staticmethod
    def create():
        net = VotingNet()

        state = [torch.load(f'assets/best_model{i}.ckp') for i in range(5)]
        for n, s in zip(net.voters, state):
            n.load_state_dict(s)


        return net

    def forward(self, x):

        y = [net(x)[:, None, :] for net in self.voters]
        x = torch.cat(y, dim=1)
        print('after cat', x.shape)
        conf = torch.maximum(x, 1-x)
        max_ix = torch.argmax(conf, dim=1)
        print('max_ix', max_ix.shape)
        mask = torch.nn.functional.one_hot(max_ix)
        mask = torch.transpose(max_ix, 1, 2)
        print('mask', mask.shape)
        mask = mask > 0.5
        x = x[mask]
        x = x.view(mask.size(0), mask.size(1))

#        x = torch.mean(x, dim=1)
#        x = self.sig((x-0.5)*2)
        return x
