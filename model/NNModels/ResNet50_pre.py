import numpy as np
import torch.nn

import torchvision as tv

class ResNet50_Pretrained(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = tv.models.resnet50(
            weights=tv.models.ResNet50_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = torch.nn.Linear(2048, 512)
        self.dropout = torch.nn.Dropout(inplace=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc = torch.nn.Linear(512, 2)
        self.sigmoid = torch.nn.Sigmoid()

        self.feature_learning = True

    def train(self, train=True):
        super().train(train)

        if not self.feature_learning:
            self.model.layer4.eval()

    def set_epoch(self, epoch):
        '''
        if int(epoch/10) % 2 == 1:
            if not self.feature_learning:
                for param in self.model.layer4.parameters():
                    param.requires_grad = True
                self.feature_learning = True
        else:
            if self.feature_learning:
                for param in self.model.layer4.parameters():
                    param.requires_grad = False
                    param.grad = None
                self.feature_learning = False
        '''

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
