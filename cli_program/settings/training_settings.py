from model.NNModels.ResNet34_pre import ResNet34_Pretrained
from utils.loss_utils import AdamFactory, ASLCalculator, calc_BCE_loss, select_best_metric, calc_MSE_loss, \
    WeightedASLCalculator, SSIMCalculator
from utils.stat_tools import calc_multi_f1, calc_f1_m

# training
MAX_EPOCH = 100
BATCH_SIZE = 16
PATIENCE = 10
WINDOW = 5

# optimizer
LR = 0.0001
DECAY = 0.0001
OPTIMIZER_FACTORY = AdamFactory(decay=DECAY, lr=LR)

# loss fct
GAMMA_NEG = 4
GAMMA_POS = 1
CLIP = 0.05

#LOSS_CALCULATOR = ASLCalculator(GAMMA_NEG, GAMMA_POS, CLIP)
#LOSS_CALCULATOR = WeightedASLCalculator(GAMMA_NEG, GAMMA_POS, CLIP)
LOSS_CALCULATOR = SSIMCalculator()

TRAINING_LOSS = LOSS_CALCULATOR.calc
VALIDATION_LOSS = LOSS_CALCULATOR.calc

# metric calculation
METRIC_CALC = None#calc_f1_m
BEST_METRIC_SELECTOR = None#select_best_metric
