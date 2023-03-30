from model.NNModels.ResNet34_pre import ResNet34Sig, ResNet34SoftMax, ResNet34SigAux
from model.NNModels.ResNet50_pre2 import ResNet50Sig, ResNet50SigAux


class ModelTypes:
    ResNet50_Sig = 'resnet50_sig'
    ResNet50_SigAux = 'resnet50_sig_aux'
    ResNet34_Sig = 'resnet34_sig'
    ResNet34_SigAux = 'resnet34_sig_aux'
    ResNet34_SoftMax = 'resnet34_softmax'

class ModelFactory:
    @staticmethod
    def create(type, state, config):
        if type == ModelTypes.ResNet34_Sig:
            return ResNet34Sig(out_cnt=config['out_cnt'],
                               pre_path=config.get('pre_path'),
                               multi_layer=config['multi_layer'])
        if type == ModelTypes.ResNet34_SigAux:
            return ResNet34SigAux(out_cnt=config['out_cnt'],
                               pre_path=config.get('pre_path'),
                               multi_layer=config['multi_layer'])

        elif type == ModelTypes.ResNet50_Sig:
            return ResNet50Sig(out_cnt=config['out_cnt'],
                               pre_path=config.get('pre_path'),
                               multi_layer=config['multi_layer'])

        elif type == ModelTypes.ResNet50_SigAux:
            return ResNet50SigAux(out_cnt=config['out_cnt'],
                               pre_path=config.get('pre_path'),
                               multi_layer=config['multi_layer'])

        elif type == ModelTypes.ResNet34_MaxSoft:
            return ResNet34SoftMax(out_cnt=config['out_cnt'],
                                   pre_path=config.get('pre_path'))

        raise NotImplemented('model type not recognized')

