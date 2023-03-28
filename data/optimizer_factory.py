import torch


class OptimizerTypes:
    ADAM = 'Adam'


class OptimizerFactory:
    @staticmethod
    def create(type, state, config):
        if type == OptimizerTypes.ADAM:
            return torch.optim.Adam(state['model'].parameters(),
                                    weight_decay=config['decay'],
                                    lr=config['lr'])

        raise NotImplemented('optimizer type not recognized')


