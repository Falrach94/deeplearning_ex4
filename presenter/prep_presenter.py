import numpy as np
import pandas as pd
import torch
from PyQt6.QtCore import pyqtSlot, pyqtSignal, QObject
from skimage.color import gray2rgb
from skimage.io import imread

from gui.PreProcessingWindow import PreProcessingWindow
from model.NNModels import ResNet50v2_pre
from model.NNModels.ResNet50v2_pre import ResNet50v2_Pretrained
from model.preprocessing.preprocessor import Preprocessor

import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class PrepPresenter(QObject):

    image_changed = pyqtSignal(object, object, object)

    def __init__(self):
        super().__init__()

        data = pd.read_csv('assets/data_dif.csv', sep=';')
        self.data = np.array(data)
        self.data_selection = 0
        self.selected_image = None
        self.selected_image_label = None

        self.preprocessor = Preprocessor()

        self.model = ResNet50v2_Pretrained()
        cp_path = "assets/good_models/787_791_50v2.ckp"
        data = torch.load(cp_path, 'cuda')
        self.model.load_state_dict(data['state_dict'])

        # init gui
        self._pre_proc_window = PreProcessingWindow()
        self._pre_proc_window.signal_change_image.connect(self.on_ppw_change_image)
        self._pre_proc_window.param_changed.connect(self.on_param_changed)

        self.image_changed.connect(self._pre_proc_window.on_image_changed)

        self.change_data_selection(0)

        self._pre_proc_window.show()

    @pyqtSlot(dict)
    def on_param_changed(self, params):
        self.preprocessor.freq_fiter.sig = params['sig']
        self.preprocessor.freq_fiter.w = params['w']
        self.preprocessor.freq_fiter.d = params['d']
        self.preprocessor.freq_fiter.recalc_filter()

        self.refresh()

    @pyqtSlot(bool)
    def on_ppw_change_image(self, next):
        self.change_data_selection(1 if next else -1)

    def change_data_selection(self, iDelta):
        self.data_selection += iDelta
        if self.data_selection < 0:
            self.data_selection = len(self.data) - 1
        if self.data_selection >= len(self.data):
            self.data_selection = 0

        self.refresh()

    def refresh(self):
        item = self.data[self.data_selection]
        path = f'assets/{item[0]}'
        self.selected_image_label = (item[1], item[2])
        self.selected_image = np.array(imread(path))
        image = gray2rgb(self.selected_image)
       # transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
        #                                     tv.transforms.ToTensor(),
        #                                     tv.transforms.Normalize(train_mean, train_std)])
        transform=tv.transforms.Compose([tv.transforms.ToPILImage(),
                               tv.transforms.ToTensor(),
                               tv.transforms.Normalize(train_mean, train_std),
                               tv.transforms.GaussianBlur(7),
                               tv.transforms.RandomAutocontrast()])

        image = transform(image)
        #image = image[np.newaxis, ...]
        #pred = self.model.forward(image)
        pred=[[0, 0]]
        self.image_changed.emit([self.selected_image,
                                 image],
                                self.selected_image_label,
                                pred[0])
