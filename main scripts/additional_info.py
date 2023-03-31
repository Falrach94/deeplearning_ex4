import sys

import pandas as pd
from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QApplication
from imageio.v2 import imread
from matplotlib.backends.backend_qt import MainWindow

class Window(MainWindow):

    def __init__(self):
        super().__init__()


class Presenter(QObject):
    def __init__(self):
        super().__init__()

        self.window = Window()
        self.window.show()

        self.df = pd.read_csv('../assets/data.csv')


    def load_image(self, ix):
        path = '../assets/' + self.df.loc[ix, 'filename']
        im = imread(path)
        label = self.df.loc[ix, '']


if __name__ == '__main__':
    app = QApplication(sys.argv)
    presenter = Presenter()
    app.exec()


