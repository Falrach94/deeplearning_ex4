import numpy as np
import torch
from torchgeometry.losses import SSIM

#import torchmetrics

from model.training.losses.asl_loss import WeightedAsymmetricLossOptimized, AsymmetricLossOptimized, ASLSingleLabel


def calc_BCE_loss(input, pred, label, metrics):
    return torch.nn.functional.binary_cross_entropy(pred.float(), label.float())

def calc_MSE_loss(input, pred, label, metrics):
    return torch.nn.functional.mse_loss(pred, input[:,0:1,:,:])

#def calc_SSIM_loss(input, pred, label, metrics):
#    return torchgeometry

class SSIMCalculator:
    def __init__(self):

        mean = 0.59685254
        std = 0.16043035
        max_val = np.abs(1/std)

        self.ssim = SSIM(window_size=5, max_val=max_val, reduction='mean').cuda()

    def calc(self, input, pred, label, metrics):
        return self.ssim(pred, input[:,0:1,:,:])
        #return self.ssim(torch.mean(pred, dim=1, keepdim=True), input[:, 0:1, :, :])


class ASLCalculator:
    def __init__(self, g_n, g_p, clip):
        self.loss = AsymmetricLossOptimized(g_n, g_p, clip).cuda()

    def calc(self, input, pred, label, metrics):
        return self.loss(pred, label)
class Single_ASLCalculator:
    def __init__(self, g_n, g_p, clip):
        self.loss = ASLSingleLabel(gamma_pos=g_p, gamma_neg=g_p, eps=clip).cuda()

    def calc(self, input, pred, label, metrics):
        return self.loss(pred, label)

class WeightedASLCalculator:
    def __init__(self, g_n, g_p, clip):
        self.loss = WeightedAsymmetricLossOptimized(g_n, g_p, clip).cuda()

    def set_weights(self, w):
        self.loss.set_weights(w)

    def calc(self, input, pred, label, metrics):
        return self.loss(pred, label)

def select_best_metric(new_metric, old_metric):
    if old_metric is None or old_metric < new_metric['mean']:
        return new_metric['mean'], True
    else:
        return old_metric, False


class AdamFactory:
    def __init__(self, decay, lr):
        self.decay = decay
        self.lr = lr

    def create(self, params):
        return torch.optim.Adam(params, weight_decay=self.decay, lr=self.lr)

