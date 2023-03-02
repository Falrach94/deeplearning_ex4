import time
import numpy as np
import torch as t
import torch.cuda

from model.training.autoEncTrainer import AutoEncTrainer
from model.training.losses.asl_loss import AsymmetricLossOptimized


class AutoEncTrainerEx(AutoEncTrainer):
    def __init__(self, cf, aef, ld):
        super().__init__()

        self.image_loss = torch.nn.MSELoss()
        self.classifier_loss = AsymmetricLossOptimized(3)
        self.sparse_loss = torch.nn.L1Loss()
        #self.classifier_loss = torch.nn.BCELoss()

        self.a = aef
        self.b = cf
        self.ld = ld

    def calc_loss(self, input, pred, label):
        return self.a * self.image_loss(pred[0], input) + self.b * self.classifier_loss(pred[1], label) + self.ld*self.sparse_loss(pred[2])

