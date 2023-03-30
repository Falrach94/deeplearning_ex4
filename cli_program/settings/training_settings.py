from data.loss_factory import LossTypes
from data.optimizer_factory import OptimizerTypes
from utils.loss_utils import AdamFactory, ASLCalculator, calc_BCE_loss, select_best_metric, calc_MSE_loss, \
    WeightedASLCalculator, SSIMCalculator, Single_ASLCalculator
from utils.stat_tools import calc_multi_f1, calc_f1_m

# training
MAX_EPOCH = 100
BATCH_SIZE = 16
PATIENCE = 10
WINDOW = 5

# optimizer
OPTIMIZER_TYPE = OptimizerTypes.ADAM
LR = 0.0001
DECAY = 0.00001
OPTIMIZER_CONFIG = {'lr': LR, 'decay': DECAY}


# loss fct
TR_LOSS_TYPE = LossTypes.BCE_WEIGHTED
GAMMA_NEG = 2
GAMMA_POS = 0
CLIP = 0.1
TR_LOSS_CONFIG = {
    'gn': GAMMA_NEG,
    'gp': GAMMA_POS,
    'clip': CLIP,
    'set_type': 'tr'
}

VAL_LOSS_TYPE = LossTypes.BCE_WEIGHTED
VAL_LOSS_CONFIG = {
    'gn': GAMMA_NEG,
    'gp': GAMMA_POS,
    'clip': CLIP,
    'set_type': 'val'
}
# metric calculation
METRIC_CALC = calc_f1_m
BEST_METRIC_SELECTOR = select_best_metric
