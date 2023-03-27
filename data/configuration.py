from utils.loss_utils import AdamFactory


def adam_config(lr, decay):
    return {
        'optimizer':
        {
            'factory': AdamFactory(decay=decay, lr=lr),
            'lr': lr,
            'decay': decay
        }
    }


def early_stopping_config(patience, window, max_epoch):
    return {
        'early_stopping':
        {
            'patience': patience,
            'window': window,
            'max_epoc': max_epoch
        }
    }


def asl_config(gn, gp, clip, calculator_type):
    return {
        'loss': {
            'gamma_neg': gn,
            'gamma_pos': gp,
            'clip': clip,
            'calculator': calculator_type(gn, gp, clip)
        }

    }

def


def create_configuration(training):
    config = dict()



    return config