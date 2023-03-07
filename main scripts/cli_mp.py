import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.Datasets.autoencoder_dataset import AutoencoderDataset
from model.NNModels.MultipathResnet import MultipathResNet34
from model.NNModels.ResNet34_pre import ResNet34_Pretrained
from model.NNModels.ResNet50v2_pre import ResNet50v2_Pretrained
from model.config import WORKER_THREADS
from model.reader.kfold_reader import KFoldReader
from model.training.autoEncTrainer import AutoEncTrainer
from utils.cli_table_builder import TableBuilder
from utils.console_util import print_progress_bar, ScreenBuilder, TableBuilderEx
from utils.loss_utils import calc_MSE_loss, select_best_metric, AdamFactory, ASLCalculator
from utils.stat_tools import calc_multi_f1

# path consts
save_path = 'assets/last_state.aes'
best_model_path = 'assets/base_model'
export_path = 'assets/export'

# training
FOLDS = 5
MAX_EPOCH = 10
BATCH_SIZE = 8
PATIENCE = 10
WINDOW = 5

# model
MODEL = MultipathResNet34(FOLDS)

# metric calculation
METRIC_CALC = calc_multi_f1
SELECT_BEST_METRIC = select_best_metric

# optimizer
lr = 0.00003
decay = 0.00001
OPTIMIZER_FACTORY = AdamFactory(decay, lr)

# loss fct
gamma_neg = 3
gamma_pos = 3
clip = 0.05

loss_calculator = ASLCalculator(gamma_neg, gamma_pos, clip)

TRAINING_LOSS = loss_calculator.calc
VALIDATION_LOSS = calc_MSE_loss

# data
NORMALIZE = True
REMOVE_UNLABLED_AUGS = False

# behaviour configuration
EXPORT = True
SAVE_MODEL = True

EXPORT_BEST_METRIC = False
EXPORT_BEST_LOSS = True


sb = ScreenBuilder()


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

        self.holdout_tr_dl = None
        self.holdout_val_dl = None

        self.model_state = None
        self.optimizer = None

        self.net_eval = [None] * FOLDS
        self.ensemble_eval = ['?']*6

        # --- setup training -----
        self.initialize_training_data()
        self.initialize_model_state()

        self.start_time = 0

        self.prepare_ui()
        self.train()

    def prepare_ui(self):

        self.print_ensemble_metrics()
        print_progress_bar(f'epoch ? - training',
                           0, 1,
                           f'',
                           sb=sb, name='tr_prog')
        print_progress_bar(f'epoch ? - validation',
                           0, 1,
                           f'',
                           sb=sb, name='val_prog')


    def initialize_training_data(self):

        # create k fold data split
        sb.print_line(f'split data into {FOLDS} folds')
        reader = KFoldReader(k=FOLDS, remove_unlabled_augs=REMOVE_UNLABLED_AUGS, holdout=0.2)

        tr_dataset = [AutoencoderDataset(fold[0], 'train', 1, NORMALIZE) for fold in reader.folds]
        val_dataset = [AutoencoderDataset(fold[1], 'val', 0, NORMALIZE) for fold in reader.folds]
        self.tr_dl = [DataLoader(data, batch_size=BATCH_SIZE,
                                 num_workers=WORKER_THREADS, shuffle=True) for data in tr_dataset]
        self.val_dl = [DataLoader(data, batch_size=BATCH_SIZE,
                                  num_workers=WORKER_THREADS, shuffle=False) for data in val_dataset]
        sb.print_line('training sample cnt:', [len(tr) for tr in tr_dataset])
        sb.print_line('validation sample cnt:', [len(val) for val in val_dataset])

        self.holdout_tr_dl = DataLoader(AutoencoderDataset(reader.training_data, 'train', 1, NORMALIZE),
                                        batch_size=BATCH_SIZE, num_workers=WORKER_THREADS, shuffle=True)
        self.holdout_val_dl = DataLoader(AutoencoderDataset(reader.holdout_set, 'val', 0, NORMALIZE),
                                         batch_size=BATCH_SIZE, num_workers=WORKER_THREADS, shuffle=False)

        # create k models and optimizers
        self.model = MODEL.cuda()
        #self.optimizer = [OPTIMIZER_FACTORY.create(self.model.parameters())
        #                  for _ in range(FOLDS)]

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

    def print_ensemble_metrics(self):

        table = TableBuilderEx(sb, 'val')
        table.add_line('nbr', 'loss', 'f1-c', 'f1-i', 'f1-m', 'crack (tp, tn, fp, fn)', 'inactive (tp, tn, fp, fn)')
        for i, info in enumerate(self.net_eval):
            if info is None:
                table.add_line(i+1, '?', '?', '?', '?', '?', '?')
            else:
                table.add_line(i+1, info[0], info[1], info[2], info[3], info[4], info[5])
        table.add_line('ensemble', *self.ensemble_eval)
        table.print()

    def eval_ensemble_net(self, i):
        self.trainer.set_session(self.model, None,
                                 None, self.holdout_val_dl,
                                 BATCH_SIZE)
        self.model.set_path(i, train=False)
        loss, time, metric = self.trainer.val_test()
        self.net_eval[i] = [loss.item(),
                            metric['crack']['f1'],
                            metric['inactive']['f1'],
                            metric['mean'],
                            (metric['crack']['tp'],
                             metric['crack']['tn'],
                             metric['crack']['fp'],
                             metric['crack']['fn']),
                            (metric['inactive']['tp'],
                             metric['inactive']['tn'],
                             metric['inactive']['fp'],
                             metric['inactive']['fn'])]

        self.print_ensemble_metrics()

    def eval_ensemble(self):
        self.trainer.set_session(self.model, None,
                                 None, self.holdout_val_dl,
                                 BATCH_SIZE)
        self.model.set_path(None, train=False)
        loss, time, metric = self.trainer.val_test()
        self.ensemble_eval = [loss.item(),
                              metric['crack']['f1'],
                              metric['inactive']['f1'],
                              metric['mean']
                              (metric['crack']['tp'],
                               metric['crack']['tn'],
                               metric['crack']['fp'],
                               metric['crack']['fn']),
                              (metric['inactive']['tp'],
                               metric['inactive']['tn'],
                               metric['inactive']['fp'],
                               metric['inactive']['fn'])]
        self.print_ensemble_metrics()

    def train_ensemble_net(self, i, tr_dl, val_dl):

        self.model.set_path(i, True)

        sb.print_line(f'starting fold {i + 1} / {FOLDS}')
        self.trainer.set_session(self.model, OPTIMIZER_FACTORY.create(self.model.parameters()),
                                 tr_dl, val_dl,
                                 BATCH_SIZE)

        model_state, metric_model_state = self.trainer.train_with_early_stopping(
            max_epoch=MAX_EPOCH,
            patience=PATIENCE,
            window=WINDOW,
            best_metric_sel=SELECT_BEST_METRIC
        )
        torch.save(model_state, best_model_path+'.ckp')
        return model_state

    def train_ensemble(self):

        self.model.set_path(None, True)
        optimizer = OPTIMIZER_FACTORY.create(self.model.parameters())
        self.trainer.set_session(self.model, optimizer,
                                 self.holdout_tr_dl, self.holdout_val_dl,
                                 BATCH_SIZE)

        model_state, metric_model_state = self.trainer.train_with_early_stopping(
            max_epoch=MAX_EPOCH,
            patience=PATIENCE,
            window=WINDOW,
            best_metric_sel=SELECT_BEST_METRIC
        )
        torch.save(model_state, best_model_path+'.ckp')
        return model_state

    def train(self):
        self.start_time = time.time_ns()

        # train separate ensemble nets
        for i, (tr_dl, val_dl) in enumerate(zip(self.tr_dl, self.val_dl)):
         #   state = self.train_ensemble_net(i, tr_dl, val_dl)
         #   self.model.load_state_dict(state)
            for j in range(FOLDS):
                self.eval_ensemble_net(j)
            self.print_ensemble_metrics()

        # train ensemble
        state = self.train_ensemble()
        self.model.load_state_dict(state)
        self.eval_ensemble()

        self.export(state, export_path)


    def print_metrics(self, loss, time, metrics, best, total_time):

        builder = TableBuilderEx(sb, name='epoch')
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
        sb.print_line('starting export')

        model = MODEL

        model.load_state_dict(state)

        model.cpu()
        model.eval()

        x = torch.randn(1, 3, 300, 300, requires_grad=True)
        y = model(x)
        torch.onnx.export(model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          path + '.zip',  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                        'output': {0: 'batch_size'}})

        model.cuda()

        sb.print_line('export finished')

    def save_progress(self):
        self.model_state['state_dict'] = self.model.state_dict()
        torch.save(self.model_state, save_path)

    # --- handler ------------------------
    def batch_callback(self, batch_ix, batch_cnt, time, training):
        if batch_ix == batch_cnt:
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
                           f'~{int(approx_rem)} s remaining (~{round(tpb,2)} s/batch)',
                           sb=sb, name='tr_prog' if training else 'val_prog')

    def epoch_callback(self, epoch, loss, time, metrics, best):
#        sb.print_line(f'epoch {epoch} finished', end='\r', flush=True)
        self.metric_update(loss, time, metrics, best)


sb.print_line('batch size:', BATCH_SIZE)
sb.print_line('learning rate', lr)
sb.print_line('weight decay', decay)
sb.print_line('worker threads', WORKER_THREADS)

controller = Controller()
