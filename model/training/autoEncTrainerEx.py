import time
import numpy as np
import torch as t
import torch.cuda

from model.training.autoEncTrainer import AutoEncTrainer
from model.training.losses.asl_loss import AsymmetricLossOptimized


class AutoEncTrainerEx(AutoEncTrainer):
    def __init__(self, cf, aef, ld):
        super().__init__()

        self.image_loss = torch.nn.MSELoss().cuda()
        self.classifier_loss = AsymmetricLossOptimized(3).cuda()

        self.a = aef
        self.b = cf
        self.ld = ld

    def calc_loss(self, input, pred, label):
        return self.a * self.image_loss(pred[0], input) \
            + self.b * self.classifier_loss(pred[1], label) \
            + self.ld*torch.sum(torch.abs(pred[2]))

