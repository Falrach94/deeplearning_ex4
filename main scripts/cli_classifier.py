import os
import threading

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.Datasets.autoencoder_dataset import AutoencoderDataset
from model.NNModels.AutoEncoder import ResNetAutoEncoder
from model.NNModels.AutoEncoderClassifier import ResNet34AutoEnc
from model.config import WORKER_THREADS
from model.profiles.builder.descriptor import Descriptor
from model.profiles.builder.hyper_parameter import HyperParameter
from model.reader.small_reader import SmallDataReader
from model.training.genericTrainer import GenericTrainer
from model.training.losses.asl_loss import AsymmetricLossOptimized
from utils.console_util import print_progress_bar
from utils.stat_tools import calc_multi_f1

BATCH_SIZE = 16
lr = 0.0001
decay = 0.00003
save_path = 'assets/classifier_save.aes'
best_autoenc_path = 'assets/best_model.ckp'
best_classifier_path = 'assets/best_classifier_model.ckp'
export_path = 'assets/export'


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
        self.start_training()

    def initialize_training_data(self):

        if os.path.exists('assets/tr_data.csv'):
            tr_data = pd.read_csv('assets/tr_data.csv', sep=';')
            val_data = pd.read_csv('assets/val_data.csv', sep=';')
            print('successfully loaded old data set')
        else:
            data_reader = SmallDataReader()
            tr_data, val_data = data_reader.get_csv_data(Descriptor('', [HyperParameter('ValSplit', 'float', 0.2)]))
            tr_data.to_csv('assets/tr_data.csv', sep=';', index=False)
            val_data.to_csv('assets/val_data.csv', sep=';', index=False)
            print('no data set found; new split created')

        self.tr_dataset = AutoencoderDataset(tr_data, 'train', 1, False, True)
        self.val_dataset = AutoencoderDataset(val_data, 'val', 0, False, True)

        print('training sample cnt:', len(self.tr_dataset))
        print('validation sample cnt:', len(self.val_dataset))

        self.tr_dl = DataLoader(self.tr_dataset, batch_size=BATCH_SIZE, num_workers=WORKER_THREADS, shuffle=True)
        self.val_dl = DataLoader(self.val_dataset, batch_size=BATCH_SIZE, num_workers=WORKER_THREADS)

        autoencoder = ResNetAutoEncoder()
        best_model = torch.load(best_autoenc_path)
        autoencoder.load_state_dict(best_model)
        self.model = ResNet34AutoEnc(autoencoder)
        self.model.cuda()

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=lr,
                                     weight_decay=decay)
#        loss = AsymmetricLossOptimized()
        loss = torch.nn.BCELoss()

        self.trainer = GenericTrainer()
        self.trainer.metric_calculator = calc_multi_f1
        self.trainer.batch_callback = self.batch_callback
        self.trainer.epoch_callback = self.epoch_callback
        self.trainer.set_session(self.model, loss, optimizer, self.tr_dl, self.val_dl, BATCH_SIZE)

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


    def create_sparse_representation(self):

        print('start converting data...')
        tr_data = np.empty((BATCH_SIZE*len(self.tr_dl), 128))
        tr_label = np.empty((BATCH_SIZE*len(self.tr_dl), 2))
        val_data = np.empty((BATCH_SIZE*len(self.val_dl), 128))
        val_label = np.empty((BATCH_SIZE*len(self.val_dl), 2))
        self.model.sparse()
        self.tr_dataset.label()
        for i, (x, y) in enumerate(self.tr_dl):
            x = x.cuda()
            y = y.cuda()
            print('training batch', i+1, '/', len(self.tr_dl))
            sparse = self.model(x)
            tr_data[i*BATCH_SIZE:i*BATCH_SIZE+x.size(0), :] = sparse.cpu().detach()
            tr_label[i*BATCH_SIZE:i*BATCH_SIZE+x.size(0), :] = y.cpu()
        self.tr_dataset.autoencode()
        self.val_dataset.label()
        for i, (x, y) in enumerate(self.val_dl):
            x = x.cuda()
            y = y.cuda()
            print('validation batch', i+1, '/', len(self.tr_dl))
            sparse = self.model(x)
            val_data[i*BATCH_SIZE:i*BATCH_SIZE+x.size(0), :] = sparse.cpu().detach()
            val_label[i*BATCH_SIZE:i*BATCH_SIZE+x.size(0), :] = y.cpu()
        self.val_dataset.autoencode()
        print('saving files')

        np.save('assets/sparse_tr_data.npy', tr_data, allow_pickle=True)
        np.save('assets/sparse_tr_label.npy', tr_label, allow_pickle=True)
        np.save('assets/sparse_val_data.npy', val_data, allow_pickle=True)
        np.save('assets/sparse_val_label.npy', val_label, allow_pickle=True)

        print("save complete")

        self.model.autoencode()

    def train(self):
        print(f'start training with early stopping (max epoch: 100, patience: 10, window: 5)')
        model = self.trainer.train_with_early_stopping(100, 10, 5)
        torch.save(model, best_classifier_path)
        self.save_progress()
        self.export(model)
        self.train_thread = None

    def metric_update(self, loss, time, metrics, best):
        tr_loss = loss['train']
        val_loss = loss['val']

        self.model_state['tr_loss'].append(tr_loss)
        self.model_state['val_loss'].append(val_loss)
        self.model_state['label_metrics'].append({'crack': metrics['crack'], 'inactive': metrics['inactive']})

        print(f'epoch {self.trainer.epoch} - loss:',
              '(tr', round(tr_loss, 5),
              '| val', round(val_loss, 5), ')')
        print(f'epoch {self.trainer.epoch} - f1: ',
              '(crack', round(metrics['crack']['f1'], 4),
              '| val', round(metrics['inactive'][r'f1'], 4),
              '| mean', round(metrics['mean'], 4), ')')
        if best['epoch'] is not None:
            print(f'best epoch {best["epoch"]} - loss: {best["loss"]}')
            print(f'best epoch {best["epoch"]} - f1: ',
                  '(crack', round(best['metric']['crack']['f1'], 4),
                  '| inactive', round(best['metric']['inactive']['f1'], 4),
                  '| mean', round(best['metric']['mean'], 4), ')')

        print(f'epoch {self.trainer.epoch} - time:'
              '(total', round(time['total'], 2), 's',
              ' | tr ', round(time['train'], 2), 's',
              ' | val ', round(time['val'], 2), 's)')

    def export(self, state):
        print('staring export')
        model = ResNet34AutoEnc()
        model.load_state_dict(state)
        model.eval()

        x = torch.randn(1, 3, 300, 300, requires_grad=True)
        y = model(x)
        torch.onnx.export(model,  # model being run
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
