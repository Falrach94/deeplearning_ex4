import torch

from model.config import WORKER_THREADS
from model.Datasets.data import ChallengeDataset
from model.profiles.builder.losses import Losses
from model.profiles.builder.models import Models
from model.profiles.builder.optimizers import Optimizers


class TrainingParameters:
    def __init__(self, config, tr_data, val_data):
        self.model = Models.instantiate(config.model)
        self.loss = Losses.instantiate(config.loss)
        self.optimizer = Optimizers.instantiate(config.optimizer, self.model.parameters())

        self.train_data_set = None
        self.val_data_set = None
        self.training_dl, self.eval_dl = self.create_dataloaders(config.data,
                                                                 config.model,
                                                                 tr_data, val_data)

    def create_dataloaders(self, data_desc, model_desc, tr_data, val_data):

        class_num = model_desc.get('class cnt')
        class_num = 2 if class_num is None else class_num.get_value()
        batch_size = int(data_desc.hyperparams[0].get_value())
        transform = int(data_desc.hyperparams[2].get_value())

        self.train_data_set = ChallengeDataset(tr_data, 'train', transform, class_num)
        self.val_data_set = ChallengeDataset(val_data, 'eval', 0, class_num)

        train_data = torch.utils.data.DataLoader(self.train_data_set,
                                                 batch_size=batch_size,
                                                 num_workers=WORKER_THREADS,
                                                 shuffle=True)
        eval_data = torch.utils.data.DataLoader(self.val_data_set,
                                                batch_size=batch_size,
                                                num_workers=WORKER_THREADS)
        return train_data, eval_data

