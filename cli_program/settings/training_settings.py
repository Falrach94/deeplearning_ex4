from model.NNModels.ResNet34_pre import ResNet34_Pretrained
from utils.loss_utils import AdamFactory, ASLCalculator, calc_BCE_loss, select_best_metric
from utils.stat_tools import calc_multi_f1

# training
MAX_EPOCH = 100
BATCH_SIZE = 16
PATIENCE = 10
WINDOW = 5

# optimizer
LR = 0.00003
DECAY = 0.00001
OPTIMIZER_FACTORY = AdamFactory(DECAY, LR)

# loss fct
GAMMA_NEG = 3
GAMMA_POS = 2
CLIP = 0.05

loss_calculator = ASLCalculator(GAMMA_NEG, GAMMA_POS, CLIP)

TRAINING_LOSS = loss_calculator.calc
VALIDATION_LOSS = calc_BCE_loss


# metric calculation
METRIC_CALC = calc_multi_f1
BEST_METRIC_SELECTOR = select_best_metric
