import copy
import os
import threading
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.Datasets.autoencoder_dataset import AutoencoderDataset
from model.NNModels.AutoEncoder import ResNetAutoEncoder
from model.NNModels.AutoEncoderClassifier import ResNet34AutoEnc
from model.NNModels.MultipathResnet import MultipathResNet34
from model.NNModels.ResNet34_pre import ResNet34_Pretrained
from model.NNModels.ResNet34_pre2 import ResNet34_Pretrained2
from model.NNModels.ResNet50v2_pre import ResNet50v2_Pretrained
from model.NNModels.autoenc.ScrambledAutoEncoder import ScrambledAutoEncoder
from model.NNModels.autoenc.SkipAutoEncoder import SkipAutoEncoder
from model.config import WORKER_THREADS
from model.profiles.builder.descriptor import Descriptor
from model.profiles.builder.hyper_parameter import HyperParameter
from model.reader.kfold_reader import KFoldReader
from model.reader.small_reader import SmallDataReader
from model.training.autoEncTrainer import AutoEncTrainer
from model.training.autoEncTrainerEx import AutoEncTrainerEx
from model.training.losses.asl_loss import AsymmetricLossOptimized, WeightedAsymmetricLossOptimized
from utils.cli_table_builder import TableBuilder
from utils.console_util import print_progress_bar
from utils.loss_utils import calc_BCE_loss, calc_MSE_loss, select_best_metric, AdamFactory
from utils.stat_tools import calc_multi_f1

# path consts
save_path = 'assets/last_state.aes'
best_model_path = 'assets/best_model'
export_path = 'assets/export'

# model
MODEL = MultipathResNet34()

# training
FOLDS = 5
BATCH_SIZE = 16
PATIENCE = 20
WINDOW = 10

# metric calculation
METRIC_CALC = calc_multi_f1
SELECT_BEST_METRIC = select_best_metric

# optimizer
lr = 0.0001
decay = 0.00003
OPTIMIZER_FACTORY = AdamFactory(decay, lr)

# loss fct
gamma_neg = 3
gamma_pos = 1
clip = 0.05

TRAINING_LOSS = calc_BCE_loss
VALIDATION_LOSS = calc_MSE_loss

# data
NORMALIZE = True

# behaviour configuration
EXPORT = True
SAVE_MODEL = True


class Controller:

    # --- initialization ----------------
    def __init__(self):
        super().__init__()

        # --- declarations ------
        self.train_thread = None

        self.tr_time_per_batch = [1]
        self.val_time_per_batch = [1]

        self.trainer = None
        self.model = None
        self.val_dl = None
        self.tr_dl = None
        self.tr_dataset = None
        self.val_dataset = None

        self.model_state = None

        # --- setup training -----
        self.initialize_training_data()
        self.initialize_model_state()

        self.start_time = 0

        self.train()

    def initialize_training_data(self):

        # create k fold data split
        print(f'split data into {FOLDS} folds')
        reader = KFoldReader(k=FOLDS)
        folds = reader.folds
        tr_dataset = [AutoencoderDataset(fold[0], 'train', 1, NORMALIZE) for fold in folds]
        val_dataset = [AutoencoderDataset(fold[1], 'val', 0, NORMALIZE) for fold in folds]
        self.tr_dl = [DataLoader(data, batch_size=BATCH_SIZE,
                                 num_workers=WORKER_THREADS, shuffle=True) for data in tr_dataset]
        self.val_dl = [DataLoader(data, batch_size=BATCH_SIZE,
                                  num_workers=WORKER_THREADS, shuffle=False) for data in val_dataset]
        print('training sample cnt:', [len(tr) for tr in tr_dataset])
        print('validation sample cnt:', [len(val) for val in val_dataset])

        # create k models and optimizers
        self.models = [copy.deepcopy(MODEL).cuda() for _ in range(FOLDS)]
        self.optimizer = [OPTIMIZER_FACTORY.create(model.parameters()) for model in self.models]

        self.trainer = AutoEncTrainer()
        self.trainer.metric_calculator = METRIC_CALC
        self.trainer.batch_callback = self.batch_callback
        self.trainer.epoch_callback = self.epoch_callback
        self.trainer.loss_fct = TRAINING_LOSS
        self.trainer.val_loss_fct = VALIDATION_LOSS

    def initialize_model_state(self):

        self.model_state = {
            'state_dict': None,
            'val_loss': [],
            'tr_loss': [],
            'label_metrics': []
        }

    # --- internal methods --------------

    def train(self):
        self.start_time = time.time_ns()

        print('k-fold training')
        for i, (tr_dl, val_dl) in enumerate(zip(self.tr_dl, self.val_dl)):
            print(f'starting fold {i+1} / {FOLDS}')
            self.trainer.set_session(self.models[i], self.optimizer[i],
                                     tr_dl, val_dl,
                                     BATCH_SIZE)

            model, metric_model = self.trainer.train_with_early_stopping(
                max_epoch=100,
                patience=PATIENCE,
                window=WINDOW,
                best_metric_sel=SELECT_BEST_METRIC
            )

            torch.save(model, best_model_path + f'{i}.ckp')
            self.export(model, export_path+f'{i}')

    def print_metrics(self, loss, time, metrics, best, total_time):

        builder = TableBuilder()
        builder.add_line(f'epoch: {self.trainer.epoch}',
                         f'runtime: {total_time[0]} min {total_time[1]} sec',
                         '')
        builder.add_line(f'epoch time: {round(time["total"], 1)} s',
                         f'tr time: {round(time["train"], 1)} s',
                         f'val time: {round(time["val"], 1)} s')
        builder.new_block()
        builder.add_line(f'loss',
                         f'tr {round(loss["train"], 5)}',
                         f'val {round(loss["val"], 5)}',
                         '')
        if metrics is not None:
            builder.add_line(f'f1',
                             f'crack {round(metrics["crack"]["f1"], 4)}',
                             f'inactive {round(metrics["inactive"]["f1"], 4)}',
                             f'mean {round(metrics["mean"], 4)}')

        if best['epoch'] is not None:
            builder.new_block()
            builder.add_line(f'best epoch {best["epoch"] + 1}',
                             f'loss: {round(best["loss"], 5)}',
                             '',
                             '')
            if metrics is not None:
                builder.add_line(f'f1',
                                 f'crack {round(best["metric"]["crack"]["f1"], 4)}',
                                 f'inactive {round(best["metric"]["inactive"]["f1"], 4)}',
                                 f'mean {round(best["metric"]["mean"], 4)}')

        builder.print()

    def metric_update(self, loss, epoch_time, metrics, best):
        tr_loss = loss['train']
        val_loss = loss['val']

        self.model_state['tr_loss'].append(tr_loss)
        self.model_state['val_loss'].append(val_loss)
        if metrics is not None:
            self.model_state['label_metrics'].append({'crack': metrics['crack'],
                                                      'inactive': metrics['inactive']})

        total_time_s = int((time.time_ns() - self.start_time)/10**9)
        total_time_min = int(total_time_s / 60)
        total_time_s %= 60

        self.print_metrics(loss, epoch_time, metrics, best, (total_time_min, total_time_s))

    def export(self, state, path):
        print('starting export')

        self.model.load_state_dict(state)

        self.model.cpu()
        self.model.eval()

        x = torch.randn(1, 3, 300, 300, requires_grad=True)
        y = self.model(x)
        torch.onnx.export(self.model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          path + '.zip',  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                        'output': {0: 'batch_size'}})

        print('export finished')

    def save_progress(self):
        self.model_state['state_dict'] = self.model.state_dict()
        torch.save(self.model_state, save_path)

    # --- handler ------------------------
    def batch_callback(self, batch_ix, batch_cnt, time, training):
        if batch_ix == batch_cnt:
            print()
            return

        tpb = time/(batch_ix+1)
        if training:
            self.tr_time_per_batch.append(tpb)
            if len(self.tr_time_per_batch) > 10:
                self.tr_time_per_batch = self.tr_time_per_batch[1:11]
            tpb = np.mean(self.tr_time_per_batch)
        else:
            self.val_time_per_batch.append(tpb)
            if len(self.val_time_per_batch) > 10:
                self.val_time_per_batch = self.val_time_per_batch[1:11]
            tpb = np.mean(self.val_time_per_batch)

        approx_rem = tpb * (batch_cnt - batch_ix-1)

        print_progress_bar(f'epoch {self.trainer.epoch} - {"training" if training else "validation"}',
                           batch_ix+1, batch_cnt,
                           f'~{int(approx_rem)} s remaining (~{round(tpb,2)} s/batch)')

    def epoch_callback(self, epoch, loss, time, metrics, best):
        print(f'epoch {epoch} finished', end='\r', flush=True)
        self.metric_update(loss, time, metrics, best)
#        torch.save(self.model.state_dict(), 'assets/tmp_model.ckp')


print('batch size:', BATCH_SIZE)
print('learning rate', lr)
print('weight decay', decay)
print('worker threads', WORKER_THREADS)

controller = Controller()
