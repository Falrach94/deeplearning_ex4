import os.path
import sys
import threading

import cv2
import pandas as pd
import numpy as np
import torch
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication
from skimage.color import gray2rgb
from skimage.io import imread
from torch.utils.data import DataLoader
import torchvision as tv

from gui.autoencoder_gui import AEWindow
from model.Datasets.autoencoder_dataset import AutoencoderDataset
from model.NNModels import AutoEncoder
from model.NNModels.AutoEncoder import ResNetAutoEncoder
from model.config import WORKER_THREADS
from model.profiles.builder.descriptor import Descriptor
from model.profiles.builder.hyper_parameter import HyperParameter
from model.reader.small_reader import SmallDataReader
from model.training.autoEncTrainer import AutoEncTrainer
from utils.console_util import print_progress_bar

BATCH_SIZE = 32
lr = 0.0001
decay = 0.00003
save_path = 'assets/auto_encoder_save.aes'

class Presenter(QObject):
    signal_update_loss = pyqtSignal(object, object)
    signal_update_image = pyqtSignal(object, object, object)

    def on_window_change_image(self, delta):
        if delta != 0:
            self.selected_image_idx += delta
            if self.selected_image_idx < 0:
                self.selected_image_idx = len(self.display_data) - 1
            elif self.selected_image_idx >= len(self.display_data):
                self.selected_image_idx = 0
   #         rel_path = f'assets/{self.display_data.loc[self.selected_image_idx, "filename"]}'
   #         image = torch.tensor(cv2.imread(rel_path).transpose((2, 0, 1)))
            #        image = gray2rgb(image)
   #         self.display_image = self.display_transform(image)[None, :]
        self.start_image_update()

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

        # --- init gui ---------
        self.window = AEWindow()

        self.signal_update_loss.connect(self.window.on_loss_changed)
        self.signal_update_image.connect(self.window.on_image_changed)

        self.window.signal_change_image.connect(self.on_window_change_image)

        self.window.show()

        # --- setup training -----
        self.initialize_training_data()
        self.initialize_model_state()
        self.start_image_update()
       # self.start_training()
        #self.create_sparse_representation()

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

        self.display_data = val_data.copy()
        self.display_transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                        tv.transforms.ToTensor()])
        rel_path = f'assets/{self.display_data.loc[self.selected_image_idx, "filename"]}'
        image = torch.tensor(cv2.imread(rel_path).transpose((2, 0, 1)))
        #        image = gray2rgb(image)
        self.display_image = self.display_transform(image)[None, :]
        self.display_model = ResNetAutoEncoder()

        self.tr_dl = DataLoader(self.tr_dataset, batch_size=BATCH_SIZE, num_workers=WORKER_THREADS, shuffle=True)
        self.val_dl = DataLoader(self.val_dataset, batch_size=BATCH_SIZE, num_workers=WORKER_THREADS)

        self.model = ResNetAutoEncoder()
        self.model.cuda()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=lr,
                                     weight_decay=decay)
        loss = torch.nn.MSELoss()

        self.trainer = AutoEncTrainer()
        self.trainer.batch_callback = self.callback
        self.trainer.set_session(self.model, loss, optimizer, self.tr_dl, self.val_dl)

    def initialize_model_state(self):
        if os.path.exists(save_path):
            self.model_state = torch.load(save_path)
            self.trainer.epoch = len(self.model_state['tr_loss'])
            self.model.load_state_dict(self.model_state['state_dict'])
            self.signal_update_loss.emit(self.model_state['tr_loss'], self.model_state['val_loss'])
        else:
            self.model_state = {
                'state_dict': dict(self.model.state_dict()),
                'val_loss': [],
                'tr_loss': []
            }

    # --- interface methods -------------
    def start_image_update(self):
        if self.image_thread is not None:
            return
        self.image_thread = threading.Thread(target=self.refresh_image)
        self.image_thread.start()

    def start_training(self):
        if self.train_thread is not None:
            return
        self.train_thread = threading.Thread(target=self.train)
        self.train_thread.start()

    # --- internal methods --------------
    def load_display_image(self):
        rel_path = f'assets/{self.display_data.loc[self.selected_image_idx, "filename"]}'
        image = torch.tensor(cv2.imread(rel_path).transpose((2, 0, 1)))
        #        image = gray2rgb(image)
        image = self.display_transform(image)[None, :]
        return image

    def refresh_image(self):
        self.display_model.load_state_dict(dict(self.model_state['state_dict']))
        self.display_model.eval()
        image = self.load_display_image()
        prediction = self.display_model(image)[0]
        cracked = self.display_data.loc[self.selected_image_idx, "crack"]
        inactive = self.display_data.loc[self.selected_image_idx, "inactive"]
        self.signal_update_image.emit(image[0], prediction.detach(), (cracked, inactive))

        self.image_thread = None

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
        print('start training')
        for i in range(100):
            print(f'starting epoch {self.trainer.epoch+1}', end='\r', flush=True)
            loss, time = self.trainer.single_epoch_with_eval()
            self.metric_update(loss, time)
            self.save_progress()
            self.start_image_update()
        self.train_thread = None

    def metric_update(self, loss, time):
        tr_loss = loss['train'].item()
        val_loss = loss['val'].item()

        self.model_state['tr_loss'].append(tr_loss)
        self.model_state['val_loss'].append(val_loss)

        self.signal_update_loss.emit(self.model_state['tr_loss'],
                                     self.model_state['val_loss'])

        print(f'epoch {self.trainer.epoch} - tr_loss', round(tr_loss, 5), 'val_loss', round(val_loss, 5))
        print(f'epoch {self.trainer.epoch} - time: ('
              'total', round(time['total'], 2), 's',
              ' | tr ', round(time['train'], 2), 's',
              ' | val ', round(time['val'], 2), 's)')

    def save_progress(self):
        self.model_state['state_dict'] = dict(self.model.state_dict())
        torch.save(self.model_state, save_path)

    # --- handler ------------------------
    def callback(self, batch_ix, batch_cnt, time, training):
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    presenter = Presenter()
    app.exec()



