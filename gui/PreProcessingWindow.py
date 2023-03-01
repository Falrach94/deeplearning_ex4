from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from gui.Slider import Slider
from utils.gui_tools import add_hlayout


class PreProcessingWindow(QtWidgets.QMainWindow):

    # --- signals ------------
    signal_change_image = pyqtSignal(bool)
    param_changed = pyqtSignal(dict)
    # --- slots --------------
    def on_image_changed(self, image, label, pred):
        self.ax_image.clear()
        self.ax_image.imshow(image[0])
        self.ax_pp_image.clear()
        self.ax_pp_image.imshow(image[1][0])
        self.canvas.draw()
        self.label_cracked.setText('cracked: ' + ('yes' if label[0] == 1 else 'no'))
        self.label_inactive.setText('inactive: ' + ('yes' if label[1] == 1 else 'no'))
        self.label_cracked_pred.setText('cracked: ' + str(round(pred[0].item(), 3)))
        self.label_inactive_pred.setText('inactive: ' + str(round(pred[1].item(), 3)))

    # --- handler ------------
    def button_next_clicked(self):
        self.signal_change_image.emit(True)

    def button_prev_clicked(self):
        self.signal_change_image.emit(False)

    def slider_w_changed(self, val):
        self.param['w'] = val
        self.param_changed.emit(self.param)

    def slider_d_changed(self, val):
        self.param['d'] = val
        self.param_changed.emit(self.param)

    def slider_sig_changed(self, val):
        self.param['sig'] = val
        self.param_changed.emit(self.param)

    # -- construction --------
    def __init__(self):
        super().__init__()

        self.param = {'w':6, 'd':10, 'sig':12}

        self.canvas = None
        self.figure = Figure()
        self.ax_image = self.figure.add_subplot(1, 2, 1)
        self.ax_pp_image= self.figure.add_subplot(1, 2, 2)
        self.label_cracked = None
        self.label_inactive = None
        self.label_cracked_pred = None
        self.label_inactive_pred = None

        # declarations
        main_panel = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        self.canvas = FigureCanvasQTAgg(self.figure)
        button_next = QtWidgets.QPushButton('Next')
        button_prev = QtWidgets.QPushButton('Prev')
        self.label_cracked = QtWidgets.QLabel()
        self.label_inactive = QtWidgets.QLabel()
        self.label_cracked_pred = QtWidgets.QLabel()
        self.label_inactive_pred = QtWidgets.QLabel()

        slider_w = Slider("w", 0, 20, self.param['w'])
        slider_d = Slider("d", 0, 5000, self.param['d'], True)
        slider_sig = Slider("sig", 0, 5000, self.param['sig'], True)

        # layout
        main_layout.addWidget(self.canvas)
        next_prev_layout, _ = add_hlayout(main_layout)
        next_prev_layout.addWidget(button_prev)
        next_prev_layout.addWidget(button_next)
        main_layout.addWidget(self.label_cracked)
        main_layout.addWidget(self.label_cracked_pred)
        main_layout.addWidget(self.label_inactive)
        main_layout.addWidget(self.label_inactive_pred)
        main_layout.addWidget(slider_w)
        main_layout.addWidget(slider_d)
        main_layout.addWidget(slider_sig)

        # initialization
        self.setCentralWidget(main_panel)
        main_panel.setLayout(main_layout)
        button_next.clicked.connect(self.button_next_clicked)
        button_prev.clicked.connect(self.button_prev_clicked)
        slider_w.value_changed.connect(self.slider_w_changed)
        slider_d.value_changed.connect(self.slider_d_changed)
        slider_sig.value_changed.connect(self.slider_sig_changed)
