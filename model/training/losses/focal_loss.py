import numpy as np
import torch
import torch.nn as nn
import torchvision


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

        self.bce = nn.BCELoss()

    def forward(self, pred, label):

        alpha_t = self.alpha * label + (1 - self.alpha) * (1 - label)

        focal_weight = torch.pow(1 - (pred - np.finfo(float).eps), self.gamma)

        loss = alpha_t * focal_weight * self.bce.forward(pred, label)




        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
