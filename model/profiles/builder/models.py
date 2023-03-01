import torch
import torchvision.models.vgg

from model.NNModels.Fusion import FusionNet
from model.NNModels.ResNet import ResNet
from model.NNModels.ResNet34 import ResNet34
from model.NNModels.ResNet34Ex import ResNet34Ex
from model.NNModels.ResNet50_pre import ResNet50_Pretrained
from model.NNModels.ResNet50v2 import ResNet50v2
from model.NNModels.ResNet50v2_pre import ResNet50v2_Pretrained
from model.NNModels.ResNetBigPool import ResNetBigPool
from model.NNModels.ResNetDropout import ResNetDropout
from model.NNModels.ResNetEx import ResNetEx
from model.NNModels.ResNetExtern import ResNet18Ex, ResNet152Ex, ResNet101Ex, ResNet50Ex
from model.NNModels.ResNetMulti import ResNetMulti
from model.NNModels.ResNet_Crack import ResNetCrack
from model.NNModels.VGG19 import VGG19
from model.NNModels.VGG19_pre import VGG19_Pretrained
from model.NNModels.VotingNet import VotingNet
from model.profiles.builder.descriptor import Descriptor
from model.profiles.builder.hyper_parameter import HyperParameter

import numpy as np

class Models:

    models = ['Fusion', 'ResNet', 'ResNet34', 'ResNet50v2', 'Voting', 'ResNet50_Pre', 'ResNet50v2_Pre']

    #models = ['Voting', 'VGG19', 'ResNet50_Pre', 'VGG19_Pre', 'ResNet_c', 'ResNet', 'ResNetEx', 'ResNetBigPool', 'ResNetDropout', 'ResNet34', 'ResNet34Ex', 'ResNetForest', 'ResNetExtern']

    @staticmethod
    def get_descriptor(name):
        if name == 'ResNet':
            return Descriptor(name, [HyperParameter('class cnt', 'int',  2),
                                     HyperParameter('dropout', 'float', 0.5)])
        if name == 'ResNetEx':
            return Descriptor(name, None)
        if name == 'ResNetBigPool':
            return Descriptor(name, None)
        if name == 'ResNetDropout':
            return Descriptor(name, [HyperParameter('dropout', 'float', 0.5, 0, 1)])
        if name == 'ResNet34':
            return Descriptor(name, [HyperParameter('class cnt', 'int',  2),
                                     HyperParameter('dropout', 'float', 0.5)])
        if name == 'ResNet34Ex':
            return Descriptor(name, None)
        if name == 'ResNetExtern':
            return Descriptor(name, [HyperParameter('version', 'int', 0, 0)])
        if name == 'ResNetForest':
            return Descriptor(name, [HyperParameter('version', 'int', 0, 0)])

        return Descriptor(name, None)

    @staticmethod
    def instantiate(descriptor):
        if descriptor.name == 'Fusion':
            return FusionNet()
        if descriptor.name == 'ResNet':
            return ResNet(descriptor.get('class cnt').get_value(),
                          descriptor.get('dropout').get_value())
        if descriptor.name == 'ResNet34':
            return ResNet34(descriptor.get('class cnt').get_value(),
                            descriptor.get('dropout').get_value())
        if descriptor.name == 'ResNet50v2':
            return ResNet50v2()

        if descriptor.name == 'ResNetEx':
            return ResNetEx()
        if descriptor.name == 'ResNetBigPool':
            return ResNetBigPool()
        if descriptor.name == 'ResNetDropout':
            return ResNetDropout(descriptor.hyperparams[0].get_value())
        if descriptor.name == 'ResNet34Ex':
            return ResNet34Ex()
        if descriptor.name == 'ResNetForest':
            if descriptor.hyperparams is None:
                type = 0
            else:
                type = descriptor.hyperparams[0].get_value()
            return ResNetMulti(type)
        if descriptor.name == 'ResNetExtern':
            type = int(descriptor.hyperparams[0].get_value())
            if type == 0:
                return ResNet18Ex()
            if type == 1:
                return ResNet34Ex()
            if type == 2:
                return ResNet50Ex()
            if type == 3:
                return ResNet101Ex()
            if type == 4:
                return ResNet152Ex()
        if descriptor.name == 'VGG19':
            return VGG19()
        if descriptor.name == 'VGG19_Pre':
            return VGG19_Pretrained()
        if descriptor.name == 'ResNet50_Pre':
            return ResNet50_Pretrained()
        if descriptor.name == 'ResNet50v2_Pre':
            return ResNet50v2_Pretrained()
        if descriptor.name == 'ResNet_c':
            return ResNetCrack()
        if descriptor.name == 'Voting':
            return VotingNet.create()

        raise BaseException(f'Model name "{descriptor.name}" not recognized!')
