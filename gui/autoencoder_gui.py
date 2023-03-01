import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from gui.Slider import Slider
from utils.gui_tools import add_hlayout


class AEWindow(QtWidgets.QMainWindow):

    # --- signals ------------
    signal_change_image = pyqtSignal(int)

    # --- slots --------------
    def on_image_changed(self, input, output, label):

        self.label_cracked.setText(f'cracked: {label[0]}')
        self.label_inactive.setText(f'inactive: {label[1]}')

        input = np.array(input)
        output = np.array(output)
        dif = (input-output).transpose((1, 2, 0))
        dif = (dif - dif.min()) / (dif.max() - dif.min())
        input = (input - input.min()) / (input.max() - input.min())
        output = (output - output.min()) / (output.max() - output.min())

        self.ax_image.clear()
        self.ax_image.imshow(input.transpose((1, 2, 0)))
        self.ax_pp_image.clear()
        self.ax_pp_image.imshow(output.transpose((1, 2, 0)))
        self.ax_dif_image.clear()
        self.ax_dif_image.imshow(dif)
        self.image_canvas.draw()

       # print(input.min(), input.max())
       # print(output.min(), output.max())

    def on_loss_changed(self, tr_loss, val_loss):
        self.ax_loss.clear()
        self.ax_loss.plot(tr_loss, label='tr loss')
        self.ax_loss.plot(val_loss, label='val loss')
        self.ax_loss.legend()
        self.loss_canvas.draw()

    # --- handler ------------
    def on_next_image_click(self):
        self.signal_change_image.emit(1)
    def on_prev_image_click(self):
        self.signal_change_image.emit(-1)
    def on_refresh_image_click(self):
        self.signal_change_image.emit(0)

    # -- construction --------
    def __init__(self):
        super().__init__()

        self.canvas = None
        self.image_figure = Figure()
        self.ax_image = self.image_figure.add_subplot(1, 3, 1)
        self.ax_pp_image = self.image_figure.add_subplot(1, 3, 2)
        self.ax_dif_image = self.image_figure.add_subplot(1, 3, 3)

        self.loss_figure = Figure()
        self.ax_loss = self.loss_figure.add_subplot(1, 1, 1)

        # declarations
        main_panel = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        self.image_canvas = FigureCanvasQTAgg(self.image_figure)
        self.loss_canvas = FigureCanvasQTAgg(self.loss_figure)

        button_next = QtWidgets.QPushButton('Next')
        button_prev = QtWidgets.QPushButton('Prev')
        button_refresh = QtWidgets.QPushButton('Refresh')

        self.label_cracked = QtWidgets.QLabel('cracked: ?')
        self.label_inactive = QtWidgets.QLabel('inactive: ?')

        # layout
        main_layout.addWidget(self.image_canvas)
        label_layout, _ = add_hlayout(main_layout)
        label_layout.addWidget(self.label_cracked)
        label_layout.addWidget(self.label_inactive)
        button_layout, _ = add_hlayout(main_layout)
        button_layout.addWidget(button_prev)
        button_layout.addWidget(button_next)
        button_layout.addWidget(button_refresh)
        main_layout.addWidget(self.loss_canvas)

        # initialization
        self.setCentralWidget(main_panel)
        main_panel.setLayout(main_layout)
        button_prev.clicked.connect(self.on_prev_image_click)
        button_next.clicked.connect(self.on_next_image_click)
        button_refresh.clicked.connect(self.on_refresh_image_click)
