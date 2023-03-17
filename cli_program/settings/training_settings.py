from model.NNModels.ResNet34_pre import ResNet34_Pretrained
from utils.loss_utils import AdamFactory, ASLCalculator, calc_BCE_loss, select_best_metric, calc_MSE_loss
from utils.stat_tools import calc_multi_f1, calc_f1_m

# training
MAX_EPOCH = 100
BATCH_SIZE = 8
PATIENCE = 20
WINDOW = 10

# optimizer
LR = 0.00003
DECAY = 0.00001
OPTIMIZER_FACTORY = AdamFactory(DECAY, LR)

# loss fct
GAMMA_NEG = 3
GAMMA_POS = 2
CLIP = 0.05

LOSS_CALCULATOR = ASLCalculator(GAMMA_NEG, GAMMA_POS, CLIP)

TRAINING_LOSS = calc_BCE_loss#LOSS_CALCULATOR.calc
VALIDATION_LOSS = calc_MSE_loss


# metric calculation
METRIC_CALC = calc_f1_m
BEST_METRIC_SELECTOR = select_best_metric
