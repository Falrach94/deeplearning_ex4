import torch.nn.functional

from data.utils import get_distribution
from model.training.losses.asl_loss import AsymmetricLossOptimized, WeightedAsymmetricLossOptimized


class ASLCalculator:
    def __init__(self, g_n, g_p, clip):
        self.loss = AsymmetricLossOptimized(g_n, g_p, clip).cuda()

    def calc(self, input, pred, label, metrics):
        return self.loss(pred, label)

class ASLWeightedCalculator:
    def __init__(self, g_n, g_p, clip, weights):
        self.loss = WeightedAsymmetricLossOptimized(g_n, g_p, clip).cuda()
        self.loss.set_weights(weights)

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
    ASL = 'ASL',
    ASL_WEIGHTED = 'ASL_Weighted'


class LossFactory:

    @staticmethod
    def create(type, state, config):
        if type == LossTypes.ASL:
            return ASLCalculator(config['gn'], config['gp'], config['clip'])
        if type == LossTypes.ASL_WEIGHTED:
            set_type = config['set_type']
            if 'split' not in state['data']:
                return None
            dataset = state['data']['split'][set_type]['dataset']
            dist = torch.tensor(dataset.get_categories()).cuda()
            weights = 1/dist
            return ASLWeightedCalculator(config['gn'], config['gp'], config['clip'], weights)
        elif type == LossTypes.BCE:
            return BCECalculator()
        elif type == LossTypes.MSE:
            return MSECalculator()

        raise NotImplementedError(f'loss type {type} not recognized')

