import torch.nn.functional

from model.training.losses.asl_loss import AsymmetricLossOptimized


class ASLCalculator:
    def __init__(self, g_n, g_p, clip):
        self.loss = AsymmetricLossOptimized(g_n, g_p, clip).cuda()

    def calc(self, input, pred, label, metrics):
        return self.loss(pred, label)


class MSECalculator:
    @staticmethod
    def calc(input, pred, label, metrics):
        return torch.nn.functional.mse_loss(pred.float(), label.float())


class BCECalculator:
    @staticmethod
    def calc(input, pred, label, metrics):
        return torch.nn.functional.binary_cross_entropy(pred.float(), label.float())


class LossTypes:
    MSE = 'MSE',
    BCE = 'BCE',
    ASL = 'ASL'


class LossFactory:

    @staticmethod
    def create(type, state, config):
        if type == LossTypes.ASL:
            return ASLCalculator(config['gn'], config['gp'], config['clip'])
        elif type == LossTypes.BCE:
            return BCECalculator()
        elif type == LossTypes.MSE:
            return MSECalculator()

        raise NotImplementedError(f'loss type {type} not recognized')

