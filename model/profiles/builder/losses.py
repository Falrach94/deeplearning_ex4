import torch

from model.profiles.builder.descriptor import Descriptor
from model.profiles.builder.hyper_parameter import HyperParameter
from model.training.losses.asl_loss import AsymmetricLossOptimized, ASLSingleLabel
from model.training.losses.focal_loss import FocalLoss


class Losses:

    losses = ['Focal', 'ASL', 'MSE', 'BCE']

    @staticmethod
    def get_descriptor(name):
        if name == 'MSE':
            return Descriptor(name, None)
        if name == 'BCE':
            return Descriptor(name, None)
        if name == 'NLL':
            return Descriptor(name, None)
        if name == 'ASL':
            return Descriptor(name,  [HyperParameter('gamma_neg', 'float', 4),
                                      HyperParameter('gamma_pos', 'float', 1),
                                      HyperParameter('clip', 'float', 0.05)])
        if name == 'Focal':
            return Descriptor(name, [HyperParameter('alpha', 'float', 2),
                                     HyperParameter('gamma', 'float', 2)])
        return Descriptor(name, None)

    @staticmethod
    def instantiate(descriptor):
        if descriptor.name == 'Focal':
            return FocalLoss(descriptor.get('alpha').get_value(),
                             descriptor.get('gamma').get_value())
        if descriptor.name == 'MSE':
            return torch.nn.MSELoss()
        if descriptor.name == 'BCE':
            return torch.nn.BCELoss()
        if descriptor.name == 'NLL':
            return torch.nn.NLLLoss2d()
        if descriptor.name == 'L1':
            return torch.nn.L1Loss()
        if descriptor.name == 'ASL':
            return AsymmetricLossOptimized(descriptor.get('gamma_neg').get_value(),
                                           descriptor.get('gamma_pos').get_value(),
                                           descriptor.get('clip').get_value())
        if descriptor.name == 'ASL_S':
            return ASLSingleLabel()
        raise BaseException(f'Loss name "{descriptor.name}" not recognized!')

