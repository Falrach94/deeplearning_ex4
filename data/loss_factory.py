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
    def __init__(self, weights=None):
        if weights is None:
            self.weights = None
        else:
            self.weights = weights.cuda()

    def calc(self, input, pred, label, metrics):
       # loss = torch.nn.functional.binary_cross_entropy_with_logits(pred.float(),
       #                                                             label.float(),
       #                                                             reduction='none')
        loss = torch.nn.functional.binary_cross_entropy(pred.float(),
                                                        label.float(),
                                                        reduction='none')

        if self.weights is not None:
            #col = torch.arange(0, label.shape[1]-1)[None, :].cuda().repeat(label.shape[0], 1)
            #powers = torch.pow(2, col)
            #class_ix = torch.sum(label*powers, dim=1, dtype=torch.long)
            #w = self.weights[class_ix]
            loss *= self.weights[None, :]

        return torch.mean(loss)

class LossTypes:
    MSE = 'MSE',
    BCE = 'BCE',
    BCE_WEIGHTED = 'BCE_WEIGHTED',
    ASL = 'ASL',
    ASL_WEIGHTED = 'ASL_Weighted'


class LossFactory:

    @staticmethod
    def calc_weights(set):
        dist = torch.tensor(set.get_categories()).cuda()

        dist = torch.tensor([dist[1] + dist[3], dist[2] + dist[3]])

        beta = 0.999
        dist = (1 - torch.pow(beta, dist)) / (1 - beta)
        weights = 1 / dist
        return weights

    @staticmethod
    def create(type, state, config):
        if type == LossTypes.ASL:
            return ASLCalculator(config['gn'], config['gp'], config['clip'])
        if type == LossTypes.ASL_WEIGHTED:
            set_type = config['set_type']
            if 'split' not in state['data']:
                return None

            dataset = state['data']['split'][set_type]['dataset']
            weights = LossFactory.calc_weights(dataset)

            return ASLWeightedCalculator(config['gn'], config['gp'], config['clip'], weights)
        elif type == LossTypes.BCE:
            return BCECalculator()
        elif type == LossTypes.BCE_WEIGHTED:
            set_type = config['set_type']
            if 'split' not in state['data']:
                return None
            dataset = state['data']['split'][set_type]['dataset']
            weights = LossFactory.calc_weights(dataset)
            return BCECalculator(weights)
        elif type == LossTypes.MSE:
            return MSECalculator()

        raise NotImplementedError(f'loss type {type} not recognized')

