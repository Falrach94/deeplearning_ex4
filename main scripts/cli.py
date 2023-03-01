import os
import threading

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.Datasets.autoencoder_dataset import AutoencoderDataset
from model.NNModels.AutoEncoder import ResNetAutoEncoder
from model.config import WORKER_THREADS
from model.profiles.builder.descriptor import Descriptor
from model.profiles.builder.hyper_parameter import HyperParameter
from model.reader.small_reader import SmallDataReader
from model.training.autoEncTrainer import AutoEncTrainer
from utils.console_util import print_progress_bar

BATCH_SIZE = 16
lr = 0.0001
decay = 0.00003
save_path = 'assets/auto_encoder_save.aes'

class Controller:

    # --- initialization ----------------
    def __init__(self):
        super().__init__()

        # --- declarations ------
        self.train_thread = None
        self.image_thread = None

        self.tr_time_per_batch = [1]
        self.val_time_per_batch = [1]

        self.selected_image_idx = 0
        self.display_data = None
        self.display_transform = None
        self.display_image = None
        self.display_model = None

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

        self.tr_dataset = AutoencoderDataset(tr_data, 'train', 1)
        self.val_dataset = AutoencoderDataset(val_data, 'val', 0)

        self.tr_dl = DataLoader(self.tr_dataset, batch_size=BATCH_SIZE, num_workers=WORKER_THREADS, shuffle=True)
        self.val_dl = DataLoader(self.val_dataset, batch_size=BATCH_SIZE, num_workers=WORKER_THREADS)

        self.model = ResNetAutoEncoder()
        self.model.cuda()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=lr,
                                     weight_decay=decay)
        loss = torch.nn.MSELoss()

        self.trainer = AutoEncTrainer()
        self.trainer.batch_callback = self.batch_callback
        self.trainer.set_session(self.model, loss, optimizer, self.tr_dl, self.val_dl)

    def initialize_model_state(self):
        if os.path.exists(save_path):
            self.model_state = torch.load(save_path)
            self.trainer.epoch = len(self.model_state['tr_loss'])
            self.model.load_state_dict(self.model_state['state_dict'])
        else:
            self.model_state = {
                'state_dict': dict(self.model.state_dict()),
                'val_loss': [],
                'tr_loss': []
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
        self.trainer.epoch_callback = self.epoch_callback
        model = self.trainer.train_with_early_stopping(100, 10, 5)
        torch.save(model, 'assets/best_model.ckp')
        self.save_progress()
        self.train_thread = None

    def metric_update(self, loss, time):
        tr_loss = loss['train']
        val_loss = loss['val']

        self.model_state['tr_loss'].append(tr_loss)
        self.model_state['val_loss'].append(val_loss)

        print(f'epoch {self.trainer.epoch} - loss: (tr', round(tr_loss, 5), 'val', round(val_loss, 5),')')
        print(f'epoch {self.trainer.epoch} - time: ('
              'total', round(time['total'], 2), 's',
              ' | tr ', round(time['train'], 2), 's',
              ' | val ', round(time['val'], 2), 's)')

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

    def epoch_callback(self, epoch, loss, time):
        print(f'epoch {epoch} finished', end='\r', flush=True)
        self.metric_update(loss, time)

controller = Controller()
