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
from model.NNModels.ResNet34_pre import ResNet34_Pretrained
from model.NNModels.ResNet50v2_pre import ResNet50v2_Pretrained
from model.config import WORKER_THREADS
from model.profiles.builder.descriptor import Descriptor
from model.profiles.builder.hyper_parameter import HyperParameter
from model.reader.small_reader import SmallDataReader
from model.training.autoEncTrainer import AutoEncTrainer
from model.training.autoEncTrainerEx import AutoEncTrainerEx
from model.training.losses.asl_loss import AsymmetricLossOptimized, WeightedAsymmetricLossOptimized
from utils.cli_table_builder import TableBuilder
from utils.console_util import print_progress_bar
from utils.stat_tools import calc_multi_f1


class AECLoss:
    def __init__(self, cf, aef, ld):
        self.image_loss = torch.nn.MSELoss().cuda()
        self.classifier_loss = AsymmetricLossOptimized(3).cuda()

        self.a = aef
        self.b = cf
        self.ld = ld

    def calc_loss(self, input, pred, label, metrics):
        return self.a * self.image_loss(pred[0], input) \
            + self.b * self.classifier_loss(pred[1], label) \
            + self.ld * torch.sum(torch.abs(pred[2]))

    def simple_loss(self, input, pred, label, metrics):
        return self.image_loss(pred, input)


class SimpleLoss:
    def __init__(self):
        #self.loss = AsymmetricLossOptimized(gamma_neg, gamma_pos, clip).cuda()
        self.loss = WeightedAsymmetricLossOptimized(gamma_neg, gamma_pos, clip).cuda()

    def calc_loss(self, input, pred, label, metrics):
        if metrics is None:
            weights = torch.ones(pred.size(0), 2).cuda()
        else:
            f1_c = metrics['crack']['f1']
            f1_i = metrics['inactive']['f1']
            fc = min(1/f1_c if f1_c != 0 else 3, 3)
            fi = min(1/f1_i if f1_i != 0 else 3, 3)
            weights = torch.tensor([fc, fi])
            weights = weights[None, :]
            weights = weights.repeat(pred.size(0), 1)
            weights = weights.cuda()

        return self.loss(pred, label, weights)

def calc_MSE_loss(input, pred, label, metrics):
    return torch.nn.functional.mse_loss(pred, label)


save_path = 'assets/classifier_save.aes'
best_autoenc_path = 'assets/best_model.ckp'
best_classifier_path = 'assets/best_classifier_model.ckp'
export_path = 'assets/export'

BATCH_SIZE = 16
lr = 0.0001
decay = 0.00003

gamma_neg = 4
gamma_pos = 1
clip = 0.15

PATIENCE = 20
WINDOW = 10

AUTO_FCT = 0.3
CLASS_FCT = 0.8
SPARSE_FCT = 0.5

#main_model = ResNetAutoEncoder()
main_model = ResNet34_Pretrained()

NORMALIZE = True

loss_calculator = SimpleLoss()
TRAINING_LOSS = loss_calculator.calc_loss
VALIDATION_LOSS = calc_MSE_loss


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

        self.start_training()

    def initialize_training_data(self):

        if os.path.exists('assets/tr_data.csv'):
            tr_data = pd.read_csv('assets/tr_data.csv', sep=';')
            val_data = pd.read_csv('assets/val_data.csv', sep=';')
            print('successfully loaded old data set')
        else:
            data_reader = SmallDataReader(False)
            tr_data, val_data = data_reader.get_csv_data(Descriptor('', [HyperParameter('ValSplit', 'float', 0.2)]))
            tr_data.to_csv('assets/tr_data.csv', sep=';', index=False)
            val_data.to_csv('assets/val_data.csv', sep=';', index=False)
            print('no data set found; new split created')

        self.tr_dataset = AutoencoderDataset(tr_data, 'train', 1, NORMALIZE)
        self.val_dataset = AutoencoderDataset(val_data, 'val', 0, NORMALIZE)

        print('training sample cnt:', len(self.tr_dataset))
        print('validation sample cnt:', len(self.val_dataset))

        self.tr_dl = DataLoader(self.tr_dataset, batch_size=BATCH_SIZE, num_workers=WORKER_THREADS, shuffle=True)
        self.val_dl = DataLoader(self.val_dataset, batch_size=BATCH_SIZE, num_workers=WORKER_THREADS)

        self.model = main_model
        self.model.cuda()

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=lr,
                                     weight_decay=decay)

        self.trainer = AutoEncTrainerEx(cf=CLASS_FCT, aef=AUTO_FCT, ld=SPARSE_FCT)
        self.trainer.metric_calculator = calc_multi_f1
        self.trainer.batch_callback = self.batch_callback
        self.trainer.epoch_callback = self.epoch_callback
        self.trainer.loss_fct = TRAINING_LOSS
        self.trainer.val_loss_fct = VALIDATION_LOSS
        self.trainer.set_session(self.model, optimizer, self.tr_dl, self.val_dl, BATCH_SIZE)

    def initialize_model_state(self):

        self.model_state = {
            'state_dict': dict(self.model.state_dict()),
            'val_loss': [],
            'tr_loss': [],
            'label_metrics': []
        }

    # --- interface methods -------------

    def start_training(self):
        if self.train_thread is not None:
            return
        self.train_thread = threading.Thread(target=self.train)
        self.train_thread.start()

    # --- internal methods --------------

    def select_best_metric(self, new_metric, old_metric):
        if old_metric is None or old_metric < new_metric['mean']:
            return new_metric['mean'], True
        else:
            return old_metric, False


    def train(self):
        self.start_time = time.time_ns()

        print(f'start training with early stopping (max epoch: 100, patience: {PATIENCE}, window: {WINDOW})')
        model, metric_model = self.trainer.train_with_early_stopping(100, PATIENCE, WINDOW, best_metric=self.select_best_metric)
        torch.save(model, best_classifier_path)
        self.save_progress()
        self.export(model, export_path+'_loss')
        if metric_model is not None:
            self.export(metric_model, export_path+'_metric')
        self.train_thread = None

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
        self.model_state['label_metrics'].append({'crack': metrics['crack'], 'inactive': metrics['inactive']})

        total_time_s = int((time.time_ns() - self.start_time)/10**9)
        total_time_min = int(total_time_s / 60)
        total_time_s %= 60

        self.print_metrics(loss, epoch_time, metrics, best, (total_time_min, total_time_s))

    def export(self, state):
        print('staring export')

        self.model.cpu()
        self.model.eval()

        x = torch.randn(1, 3, 300, 300, requires_grad=True)
        y = self.model(x)
        torch.onnx.export(self.model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          export_path + '.zip',  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                        'output': {0: 'batch_size'}})

        print('export finished')

    def save_progress(self):
        self.model_state['state_dict'] = dict(self.model.state_dict())
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


print('training classifier on pretrained autoencoder-enocder stage')
print('batch size:', BATCH_SIZE)
print('learning rate', lr)
print('weight decay', decay)
print('worker threads', WORKER_THREADS)

controller = Controller()
