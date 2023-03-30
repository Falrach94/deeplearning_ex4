from cli_program.settings.behaviour_settings import *
from cli_program.settings.data_settings import *
from cli_program.settings.model_settings import MODEL_TYPE, MODEL_CONFIG
from cli_program.settings.training_settings import *


def create_empty_config():
    config = dict()

    config['model'] = dict()

    config['behaviour'] = dict()

    config['data'] = dict()
    config['data']['csv'] = dict()
    config['data']['fuser'] = dict()
    config['data']['labeler'] = dict()
    config['data']['extra_labeler'] = dict()
    config['data']['augmenter'] = dict()
    config['data']['image_loader'] = dict()
    config['data']['filter'] = dict()
    config['data']['filter']['tr'] = dict()
    config['data']['filter']['val'] = dict()
    config['data']['transform'] = dict()

    config['training'] = dict()
    config['training']['metric'] = dict()
    config['training']['config'] = dict()
    config['training']['optimizer'] = dict()
    config['training']['loss'] = dict()
    config['training']['loss']['tr'] = dict()
    config['training']['loss']['val'] = dict()
    config['training']['loss']['aux'] = dict()

    config['path'] = dict()

    return config


def default_config():
    config = create_empty_config()

    config['behaviour']['mode'] = MODE
    config['behaviour']['config'] = MODE_CONFIG

    config['path']['ckp'] = BEST_MODEL_PATH
    config['path']['export'] = EXPORT_PATH

    config['model']['type'] = MODEL_TYPE
    config['model']['config'] = MODEL_CONFIG

    config['data']['csv']['path'] = DATA_PATH
    config['data']['csv']['seperator'] = CSV_SEPERATOR
    config['data']['csv']['extra_path'] = EXTRA_PATH

    config['data']['transform']['tr'] = TR_TRANSFORMS
    config['data']['transform']['val'] = VAL_TRANSFORMS

    config['data']['labeler']['type'] = LABELER_TYPE
    config['data']['labeler']['config'] = LABELER_CONFIG

    config['data']['extra_labeler']['type'] = EXTRA_LABELER_TYPE
    config['data']['extra_labeler']['config'] = EXTRA_LABELER_CONFIG

    config['data']['fuser']['type'] = FUSER_TYPE
    config['data']['fuser']['config'] = FUSER_CONFIG

    config['data']['augmenter']['type'] = AUGMENTER_TYPE
    config['data']['augmenter']['config'] = AUGMENTER_CONFIG

    config['data']['filter']['tr']['type'] = TR_FILTER_TYPE
    config['data']['filter']['tr']['config'] = TR_FILTER_CONFIG
    config['data']['filter']['val']['type'] = VAL_FILTER_TYPE
    config['data']['filter']['val']['config'] = VAL_FILTER_CONFIG

    config['data']['image_loader']['type'] = IMAGE_LOADER_TYPE
    config['data']['image_loader']['config'] = IMAGE_LOADER_CONFIG

    config['data']['split'] = HOLDOUT_SPLIT

    config['training']['optimizer']['type'] = OPTIMIZER_TYPE
    config['training']['optimizer']['config'] = OPTIMIZER_CONFIG

    config['training']['loss']['tr']['type'] = TR_LOSS_TYPE
    config['training']['loss']['tr']['config'] = TR_LOSS_CONFIG
    config['training']['loss']['val']['type'] = VAL_LOSS_TYPE
    config['training']['loss']['val']['config'] = VAL_LOSS_CONFIG
    config['training']['loss']['aux']['type'] = AUX_LOSS_TYPE
    config['training']['loss']['aux']['config'] = AUX_LOSS_CONFIG

    config['training']['config']['max_epoch'] = MAX_EPOCH
    config['training']['config']['patience'] = PATIENCE
    config['training']['config']['window'] = WINDOW
    config['training']['config']['batch_size'] = BATCH_SIZE

    config['training']['metric']['calculator'] = METRIC_CALC
    config['training']['metric']['selector'] = BEST_METRIC_SELECTOR
    return config

