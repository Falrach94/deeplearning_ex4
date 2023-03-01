import os.path

import numpy as np
import pandas as pd

from model.profiles.builder.data_readers import DataReaders
from model.reader.data_reader import DataReader
from model.training.Checkpoint import Checkpoint
from model.training.training_parameters import TrainingParameters
from utils.json_util import dic_to_json, serlist_to_json


class Session:
    def __init__(self,
                 name,
                 training_loss=None,
                 eval_loss=None,
                 f1_m=None,
                 f1_c=None,
                 f1_i=None,
                 best_f1m=[0, 0, 0],
                 best_f1c=[0, 0, 0],
                 best_f1i=[0, 0, 0],
                 epoch_time=None):

        self.name = name

        self.epoch_time = [] if epoch_time is None else epoch_time

        self.training_loss = [] if training_loss is None else training_loss
        self.eval_loss = [] if eval_loss is None else eval_loss

        self.f1 = [] if f1_m is None else f1_m
        self.f1_crack = [] if f1_c is None else f1_c
        self.f1_inactive = [] if f1_i is None else f1_i

        self.best_f1_i = best_f1i
        self.best_f1_c = best_f1c
        self.best_f1_m = best_f1m

        self.cp_best_f1_i = Checkpoint(f'{self.name}_best_f1_i')
        self.cp_best_f1_c = Checkpoint(f'{self.name}_best_f1_c')
        self.cp_best_f1_m = Checkpoint(f'{self.name}_best_f1_m')

        self.checkpoint = Checkpoint(self.name+'_checkpoint')

        self.tr_data = None
        self.val_data = None

        self.batch_size = None

        self.params = None

    def clone_data(self, tr_path, val_path):
        self.tr_data = pd.DataFrame(np.load(tr_path, allow_pickle=True),
                               columns=['filename', 'crack', 'inactive'])
        self.val_data = pd.DataFrame(np.load(val_path, allow_pickle=True),
                                columns=['filename', 'crack', 'inactive'])
        val_path = f'assets/datasets/{self.name}_val.npy'
        tr_path = f'assets/datasets/{self.name}_tr.npy'
        np.save(tr_path, np.array(self.tr_data), allow_pickle=True)
        np.save(val_path, np.array(self.val_data), allow_pickle=True)

    def load_or_create_data(self, config):
        val_path = f'assets/datasets/{self.name}_val.npy'
        tr_path = f'assets/datasets/{self.name}_tr.npy'
        if os.path.exists(tr_path):
            self.tr_data = pd.DataFrame(np.load(tr_path, allow_pickle=True),
                                   columns=['filename', 'crack', 'inactive'])
            self.val_data = pd.DataFrame(np.load(val_path, allow_pickle=True),
                                    columns=['filename', 'crack', 'inactive'])
        else:
            reader = DataReaders.instantiate(config.data)
            self.tr_data, self.val_data = reader.get_csv_data(config.data)
            np.save(tr_path, np.array(self.tr_data), allow_pickle=True)
            np.save(val_path, np.array(self.val_data), allow_pickle=True)

    def reset(self, cp_data=None):
        self.epoch_time = []

        self.training_loss = []
        self.eval_loss = []

        self.f1 = []
        self.f1_crack = []
        self.f1_inactive = []

        self.best_f1_i = [0, 0, 0]
        self.best_f1_c = [0, 0, 0]
        self.best_f1_m = [0, 0, 0]

        self.params = None

        self.checkpoint.reset(cp_data)


    def epoch_cnt(self):
        return len(self.f1)

    def add_epoch(self, training_loss, eval_loss, epoch_time, cp_data, f1_m, f1_c, f1_i):
        self.epoch_time.append(epoch_time)

        self.training_loss.append(training_loss)
        self.eval_loss.append(eval_loss)

        self.f1.append(f1_m)
        self.f1_crack.append(f1_c)
        self.f1_inactive.append(f1_i)

        self.checkpoint.update(cp_data)

        if f1_c > self.best_f1_c[0]:
            self.best_f1_c = [f1_c, f1_i, f1_m]
            self.cp_best_f1_c.update(cp_data)
        if f1_i > self.best_f1_i[1]:
            self.best_f1_i = [f1_c, f1_i, f1_m]
            self.cp_best_f1_i.update(cp_data)
        if f1_m > self.best_f1_m[2]:
            self.best_f1_m = [f1_c, f1_i, f1_m]
            self.cp_best_f1_m.update(cp_data)

    ### - create training parameters from config
    ### - load last checkpoint if possible
    ### - create and save or load existing training/validation data
    def initialize(self, config):
        if self.params is None:
            self.batch_size = int(config.data.get('BatchSize').get_value())
            self.load_or_create_data(config)
            self.params = TrainingParameters(config, self.tr_data, self.val_data)

            if self.epoch_cnt() > 0:

                if self.checkpoint.is_valid() and self.checkpoint.load():
                    self.params.model.load_state_dict(self.checkpoint.data['state_dict'])
                else:
                    print("Loading checkpoint failed!")
                    self.reset()
            elif self.checkpoint.is_loaded():
                self.params.model.load_state_dict(self.checkpoint.data['state_dict'])

    def get_name(self):
        return f'# {self.epoch_cnt()}'

    def to_json(self):
        tmp = dic_to_json(['name', 'training_loss', 'eval_loss', 'epoch_time', 'f1', 'f1_c', 'f1_i', 'f1_m_best', 'f1_i_best', 'f1_c_best'],
                          [self.name, self.training_loss, self.eval_loss, self.epoch_time, self.f1, self.f1_crack, self.f1_inactive, self.best_f1_m, self.best_f1_i, self.best_f1_c])

        return '{' + tmp + '}'

    @staticmethod
    def from_json(dic):
        training_loss = dic['training_loss']
        eval_loss = dic['eval_loss']
        epoch_time = dic['epoch_time']
        f1_m = dic['f1']
        f1_c = dic['f1_c']
        f1_i = dic['f1_i']
        best_f1_m = dic['f1_m_best']
        best_f1_i = dic['f1_i_best']
        best_f1_c = dic['f1_c_best']
        name = dic['name']

        return Session(name,
                       training_loss,
                       eval_loss,
                       f1_m, f1_c, f1_i,
                       best_f1_m, best_f1_c, best_f1_i,
                       epoch_time)



