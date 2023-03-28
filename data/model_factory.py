from model.NNModels.ResNet34_pre import ResNet34Sig, ResNet34SoftMax


class ModelTypes:
    ResNet34_Sig = 'resnet34_sig'
    ResNet34_SoftMax = 'resnet34_softmax'

class ModelFactory:
    @staticmethod
    def create(type, state, config):
        if type == ModelTypes.ResNet34_Sig:
            return ResNet34Sig(out_cnt=config['out_cnt'],
                               pre_path=config.get('pre_path'),
                               multi_layer=config['multi_layer'])

        elif type == ModelTypes.ResNet34_MaxSoft:
            return ResNet34SoftMax(out_cnt=config['out_cnt'],
                                   pre_path=config.get('pre_path'))

        raise NotImplemented('model type not recognized')

