import torch

from model.profiles.builder.descriptor import Descriptor
from model.profiles.builder.hyper_parameter import HyperParameter


class Optimizers:

    optimizers = ['Adam', 'AdaGrad', 'AdaDelta']

    @staticmethod
    def get_descriptor(name):
        if name == 'Adam':
            return Descriptor('Adam', [HyperParameter('learning rate', 'float', 3e-3, 0),
                                       HyperParameter('weight decay', 'float', 0.001, 0)])
        if name == 'AdaGrad':
            return Descriptor('AdaGrad', [HyperParameter('learning rate', 'float', 3e-3, 0)])
        if name == 'AdaDelta':
            return Descriptor('AdaDelta', [])
        raise BaseException(f'Optimizer name "{name}" not recognized!')

    @staticmethod
    def instantiate(descriptor, params):
        if descriptor.name == 'Adam':
            return torch.optim.Adam(params,
                                    lr=descriptor.get('learning rate').get_value(),
                                    weight_decay=descriptor.get('weight decay').get_value())
        if descriptor.name == 'AdaGrad':
            return torch.optim.Adagrad(params, lr=descriptor.hyperparams[0].get_value())
        if descriptor.name == 'AdaDelta':
            return torch.optim.Adadelta(params)
        raise BaseException(f'Optimizer name "{descriptor.name}" not recognized!')
