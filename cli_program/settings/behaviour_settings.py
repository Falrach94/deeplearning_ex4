import numpy as np

from cli_program.config_updater import ConfigUpdateField

DIST_PATH = 'assets/dist_model.ckp'
DEF_PATH = 'assets/def_model.ckp'


BEST_MODEL_PATH = DIST_PATH #'assets/best_model.ckp'
BASE_MODEL_PATH = 'assets/base_model.ckp'
EXPORT_PATH = 'assets/export'


class Modes:
    Split = 'split'
    KFold = 'kfold'



K = 5

CONFIG_UPDATES = [
    ConfigUpdateField(['training', 'loss', 'tr', 'config', 'gn'], [2])
]

MODE_CONFIG = {
    'k': K,
    'updates': CONFIG_UPDATES
}
MODE = Modes.KFold

