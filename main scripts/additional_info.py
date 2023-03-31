import sys

import pandas as pd
from PyQt6 import QtWidgets
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication
from imageio.v2 import imread
from matplotlib.backends.backend_qt import MainWindow
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from utils.gui_tools import add_hlayout


class Window(MainWindow):

    signal_change_im = pyqtSignal(int)

    def load_image(self, image, label, label_new, name):
        self.ax.clear()
        self.ax.imshow(image)
        self.canvas.draw()

        self.label1.setText(f'crack: {label[1] == 1}')
        self.label2.setText(f'inactive: {label[0] == 1}')

        self.name.setText(name)

        self.cb1.setChecked(label_new['dirty'])
        self.cb2.setChecked(label_new['line'])

    def __init__(self):
        super().__init__()

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.subplots()

        main_panel = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()

        main_layout.addWidget(self.canvas)

        label_layout, panel = add_hlayout(main_layout)
        self.label1 = QtWidgets.QLabel('crack: ')
        self.label2 = QtWidgets.QLabel('inactive: ')
        self.name = QtWidgets.QLabel('title: ')
        label_layout.addWidget(self.label1)
        label_layout.addWidget(self.label2)
        label_layout.addWidget(self.name)
        panel.setFixedHeight(50)

        add_layout, panel = add_hlayout(main_layout)
        self.cb1 = QtWidgets.QCheckBox('dirty')
        self.cb2 = QtWidgets.QCheckBox('lines')
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
        self.df['dirty'] = 0
        self.df['line'] = 0

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
        label_new = {'dirty': self.df.loc[ix, 'dirty'], 'line': self.df.loc[ix, 'line']}
        self.image_sig.emit(im, label, label_new, name)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    presenter = Presenter()
    app.exec()


