import os

import numpy as np
import torch
import torch.nn as nn

from model.NNModels.ResNet import ResNet
from model.preprocessing.freq_filter import FrequencyFilter


class VotingNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.net1 = ResNet()
        self.net2 = ResNet()
        self.net3 = ResNet()
        self.net4 = ResNet()

        self.flatten = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(8, 128)
        self.fc2 = torch.nn.Linear(128, 2)
        self.relu = torch.nn.ReLU(inplace=True)

        self.sig = torch.nn.Sigmoid()

    def load_params(self, params):
        self.net1.load_state_dict(params[0]['state_dict'])
        self.net2.load_state_dict(params[1]['state_dict'])
        self.net3.load_state_dict(params[2]['state_dict'])
        self.net4.load_state_dict(params[3]['state_dict'])

        for param in self.net1.parameters():
            param.requires_grad = False
        for param in self.net2.parameters():
            param.requires_grad = False
        for param in self.net3.parameters():
            param.requires_grad = False
        for param in self.net4.parameters():
            param.requires_grad = False

    @staticmethod
    def create():
        net = VotingNet()

        a = torch.load('assets/voters/A.ckp', 'cuda')
        b = torch.load('assets/voters/B.ckp', 'cuda')
        c = torch.load('assets/voters/C.ckp', 'cuda')
        d = torch.load('assets/voters/D.ckp', 'cuda')

        net.load_params([a,b,c,d])

        return net

    def forward(self, x):

        y1 = self.net1(x)[:, None, ...]
        y2 = self.net2(x)[:, None, ...]
        y3 = self.net3(x)[:, None, ...]
        y4 = self.net4(x)[:, None, ...]

        x = torch.cat((y1, y2, y3, y4), dim=1)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        #x = self.sig(x)

        return x

#        x = self.sig(x)

 #       return x
