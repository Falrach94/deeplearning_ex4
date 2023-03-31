import sys

import cv2
import pandas as pd
import torch
import torchvision
from PyQt6 import QtWidgets
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication
from imageio.v2 import imread
from matplotlib.backends.backend_qt import MainWindow
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from torchvision.transforms import Compose

from utils.gui_tools import add_hlayout


class Window(MainWindow):

    signal_change_im = pyqtSignal(int)

    def load_image(self, image, label, label_new, name):

        ks = 5
        padding = ks // 2
        var = 1
        self.gauss = torch.tensor(cv2.getGaussianKernel(ks, var), dtype=torch.float)[:,0]

        t = Compose([torchvision.transforms.ToPILImage(),
                     torchvision.transforms.ToTensor()
                     ])
                    # torchvision.transforms.GaussianBlur(5, sigma=(0.1, 2.0))])


        oi = image
        image = t(image)
        image = image[None, :, : ,:]

        image = torch.nn.functional.conv2d(image, weight=self.gauss.view(1, 1, -1, 1), padding=(padding, 0))
        image = torch.nn.functional.conv2d(image, weight=self.gauss.view(1, 1, 1, -1), padding=(0, padding))

        image = image.view(300, 300)

        self.ax.clear()
        self.ax2.clear()
        self.ax.imshow(oi)
        self.ax2.imshow(image)
        self.canvas.draw()

        self.label1.setChecked(label[1])
        self.label2.setChecked(label[0])
#        self.label1.setText(f'crack: {label[1] == 1}')
#        self.label2.setText(f'inactive: {label[0] == 1}')

        self.name.setText(name)

        self.cb1.setChecked(label_new['crack'])
        self.cb2.setChecked(label_new['inactive'])

    def __init__(self):
        super().__init__()

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax, self.ax2 = self.figure.subplots(1,2)

        main_panel = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()

        main_layout.addWidget(self.canvas)

        label_layout, panel = add_hlayout(main_layout)
        self.label1 = QtWidgets.QCheckBox('crack')
        self.label2 = QtWidgets.QCheckBox('inactive')
        self.name = QtWidgets.QLabel('title: ')
        label_layout.addWidget(self.label1)
        label_layout.addWidget(self.label2)
        label_layout.addWidget(self.name)
        panel.setFixedHeight(50)

        add_layout, panel = add_hlayout(main_layout)
        self.cb1 = QtWidgets.QCheckBox('pred crack')
        self.cb2 = QtWidgets.QCheckBox('pred inactive')
        add_layout.addWidget(self.cb1)
        add_layout.addWidget(self.cb2)
        panel.setFixedHeight(50)

        button_layout, panel = add_hlayout(main_layout)
        panel.setFixedHeight(50)

        self.setCentralWidget(main_panel)
        main_panel.setLayout(main_layout)
        self.prev = QtWidgets.QPushButton('prev')
        self.next = QtWidgets.QPushButton('next')
        button_layout.addWidget(self.prev)
        button_layout.addWidget(self.next)

        self.next.clicked.connect(self.emit_next)
        self.prev.clicked.connect(self.emit_prev)

    def emit_next(self):
        self.signal_change_im.emit(1)
    def emit_prev(self):
        self.signal_change_im.emit(-1)


class Presenter(QObject):

    image_sig = pyqtSignal(object, object, object, str)

    def __init__(self):
        super().__init__()

        self.window = Window()
        self.window.show()

        self.df = pd.read_csv('../assets/data.csv').reset_index()

        stats = torch.load('../assets/stats.stats')
        pred = stats['metric'][-1]['predictions'] > 0.5
        label = stats['metric'][-1]['label'] > 0.5

        df = pd.DataFrame(columns=['idx'], data=stats['metric'][-1]['idx'])
        df['idx'] = df['idx'].astype(int)
        df['p_c'] = pred[:, 0]
        df['p_i'] = pred[:, 1]
        df['crack'] = label[:, 0]
        df['inactive'] = label[:, 1]
        df['filename'] = df['idx'].astype(str)
        df['filename'] = ('000' + df['filename']).str[-4:]
        df['filename'] = 'images/cell' + df['filename'] + '.png'

        self.df = df

        self.df['dirty'] = 0
        self.df['line'] = 0

        sel = pred != label
        sel = torch.sum(sel, dim=1)
        sel = [v.item() for v in sel != 0]
        self.df = self.df[sel].reset_index(drop=True)

        self.image_sig.connect(self.window.load_image)
        self.window.signal_change_im.connect(self.change_ix)

        self.cur_ix = 0
        self.load_image(self.cur_ix)

    def change_ix(self, dir):
        self.cur_ix += dir
        if self.cur_ix == -1:
            self.cur_ix = len(self.df)-1
        if self.cur_ix == len(self.df):
            self.cur_ix = 0

        self.load_image(self.cur_ix)

    def load_image(self, ix):
        path = '../assets/' + self.df.loc[ix, 'filename']
        im = imread(path)
        name = self.df.loc[ix, 'filename'][-8:-4]
        label = [self.df.loc[ix, 'inactive'], self.df.loc[ix, 'crack']]
        label_new = {'inactive': self.df.loc[ix, 'p_i'], 'crack': self.df.loc[ix, 'p_c']}
        self.image_sig.emit(im, label, label_new, name)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    presenter = Presenter()
    app.exec()


