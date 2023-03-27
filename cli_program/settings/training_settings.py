from utils.loss_utils import AdamFactory, ASLCalculator, calc_BCE_loss, select_best_metric, calc_MSE_loss, \
    WeightedASLCalculator, SSIMCalculator, Single_ASLCalculator
from utils.stat_tools import calc_multi_f1, calc_f1_m

# training
MAX_EPOCH = 100
BATCH_SIZE = 16
PATIENCE = 10
WINDOW = 10

# optimizer
LR = 0.00003
DECAY = 0.00001
OPTIMIZER_FACTORY = AdamFactory(decay=DECAY, lr=LR)

# loss fct
GAMMA_NEG = 4
GAMMA_POS = 1
CLIP = 0.1

LOSS_CALCULATOR = ASLCalculator(GAMMA_NEG, GAMMA_POS, CLIP)
#LOSS_CALCULATOR = Single_ASLCalculator(GAMMA_NEG, GAMMA_POS, CLIP)
#LOSS_CALCULATOR = WeightedASLCalculator(GAMMA_NEG, GAMMA_POS, CLIP)
#LOSS_CALCULATOR = SSIMCalculator()

#TRAINING_LOSS = calc_BCE_loss
TRAINING_LOSS = LOSS_CALCULATOR.calc
#VALIDATION_LOSS = calc_BCE_loss
VALIDATION_LOSS = LOSS_CALCULATOR.calc

# metric calculation
METRIC_CALC = calc_f1_m
BEST_METRIC_SELECTOR = select_best_metric
