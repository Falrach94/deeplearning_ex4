import torch

from model.training.losses.asl_loss import WeightedAsymmetricLossOptimized, AsymmetricLossOptimized


def calc_BCE_loss(input, pred, label, metrics):
    return torch.nn.functional.binary_cross_entropy(pred, label)

def calc_MSE_loss(input, pred, label, metrics):
    return torch.nn.functional.mse_loss(pred, label)

class ASLCalculator:
    def __init__(self, g_n, g_p, clip):
        self.loss = AsymmetricLossOptimized(g_n, g_p, clip).cuda()

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

