from cli_program.settings.behaviour_settings import Modes
from data.augment_fuser import FuserFactory
from data.augment_generator import AugmenterFactory
from data.data_filter import FilterFactory
from data.data_reader import CSVReader
from data.dataset_generator import create_dataset, create_single_split_datasets, create_k_fold_datasets
from data.image_loader import ImageLoaderFactory
from data.label_provider import LabelerFactory
from data.loss_factory import LossFactory
from data.model_factory import ModelFactory
from data.optimizer_factory import OptimizerFactory


def initialize_model_state(config):
    model_config = config['model']
    return ModelFactory.create(model_config['type'],
                               None,
                               model_config['config'])


def initialize_training_state(state, config):
    training_state = dict()
    training_config = config['training']
    training_state['optimizer'] = OptimizerFactory.create(training_config['optimizer']['type'],
                                                          state,
                                                          training_config['optimizer']['config'])

    training_state['loss'] = dict()
    training_state['loss']['tr'] = LossFactory.create(training_config['loss']['tr']['type'],
                                                      state,
                                                      training_config['loss']['tr']['config'])

    if training_config['loss']['val']['type'] is None:
        training_state['loss']['val'] = training_state['loss']['tr']
    else:
        training_state['loss']['val'] = LossFactory.create(training_config['loss']['val']['type'],
                                                           state,
                                                           training_config['loss']['val']['config'])

    return training_state


def initialize_data_processor_state(config):
    state = dict()

    data_config = config['data']

    state['labeler'] = LabelerFactory.create(data_config['labeler']['type'],
                                             state,
                                             data_config['labeler']['config'])

    state['fuser'] = FuserFactory.create(data_config['fuser']['type'],
                                         state,
                                         data_config['fuser']['config'])

    state['augmenter'] = AugmenterFactory.create(data_config['augmenter']['type'],
                                                 state,
                                                 data_config['augmenter']['config'])

    state['image_loader'] = ImageLoaderFactory.create(data_config['image_loader']['type'],
                                                      state,
                                                      data_config['image_loader']['config'])

    state['filter'] = dict()
    state['filter']['tr'] = FilterFactory.create(data_config['filter']['tr']['type'],
                                                 state,
                                                 data_config['filter']['tr']['config'])
    state['filter']['val'] = FilterFactory.create(data_config['filter']['val']['type'],
                                                  state,
                                                  data_config['filter']['val']['config'])
    return state


def initialize_data(state, config):
    proc_state = state['data_processor']
    data_config = config['data']

    state = dict()

    # load data
    df = CSVReader(path=data_config['csv']['path'],
                   seperator=data_config['csv']['seperator']).get()

    # add labels
    labeled_df = proc_state['labeler'].label_dataframe(df)
    state['df'] = labeled_df

    # create datasets and dataloaders
    state['raw'] = create_dataset(data=labeled_df,
                                  image_provider=proc_state['image_loader'],
                                  label_provider=proc_state['labeler'],
                                  filter=None,
                                  augmenter=None,
                                  transform=data_config['transform']['val'],
                                  shuffle=False,
                                  batch_size=config['training']['config']['batch_size'])

    if config['behaviour']['mode'] == Modes.Split:
        state['split'] = create_single_split_datasets(
                data=labeled_df,
                split=data_config['split'],
                image_provider=proc_state['image_loader'],
                label_provider=proc_state['labeler'],
                augmentor=proc_state['augmenter'],
                tr_transform=data_config['transform']['tr'],
                val_transform=data_config['transform']['val'],
                batch_size=config['training']['config']['batch_size'],
                tr_filter=proc_state['filter']['tr'],
                val_filter=proc_state['filter']['val']
            )
    elif config['behaviour']['mode'] == Modes.KFold:
        state['folds'] = create_k_fold_datasets(
                k=config['behaviour']['config']['k'],
                data=labeled_df,
                image_provider=proc_state['image_loader'],
                label_provider=proc_state['labeler'],
                augmentor=proc_state['augmenter'],
                tr_transform=data_config['transform']['tr'],
                val_transform=data_config['transform']['val'],
                batch_size=config['training']['config']['batch_size'],
                tr_filter=proc_state['filter']['tr'],
                val_filter=proc_state['filter']['val'])
    else:
        raise NotImplementedError(f'mode {config["behaviour"]["mode"]} not recogizied')

    return state


def initialize_state(config):
    state = dict()
    state['model'] = initialize_model_state(config)
    state['training'] = initialize_training_state(state, config)
    state['data_processor'] = initialize_data_processor_state(config)
    state['data'] = initialize_data(state, config)
    return state

