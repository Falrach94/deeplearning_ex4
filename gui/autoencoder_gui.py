import copy

import cv2
import numpy as np
import torch
import torchmetrics
import torchvision as tv

from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from skimage import filters


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
        kernel_size = 5
        MEAN = 0.59685254
        STD = 0.16043035
        max_val = 1/STD

        ssim = torchmetrics.StructuralSimilarityIndexMeasure(gaussian_kernel=False,
                                                             kernel_size=(kernel_size, kernel_size),
                                                             reduction='none',
                                                             #data_range=max_val,
                                                             return_full_image=True)
        output = output.repeat(3, axis=0)
        input = (input - input.min()) / (input.max() - input.min())
        output = (output - output.min()) / (output.max() - output.min())


        image_tensor = torch.Tensor(input).mean(dim=0)[None, None, :, :]
        clean_tensor = torch.Tensor(output).mean(dim=0)[None, None, :, :]


       # image_tensor = tv.transforms.GaussianBlur(15)(image_tensor)

        dif = ssim(image_tensor, clean_tensor)
      #  dif = torch.abs(image_tensor - clean_tensor)[0]

#        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])[None, None,:,:].float()
#        image_tensor = torch.nn.functional.conv2d(image_tensor, kernel, padding=1)
      #  input = np.repeat(np.array(image_tensor[0]), 3, axis=0)

        #dif = np.mean(np.power(input-output, 2), axis=0)
        #dif = abs(dif - dif.min()) / (dif.max() - dif.min())
        t = 3

        dif = dif[1][0]
        dif = tv.transforms.CenterCrop((300, 300))(dif)
        #dif = tv.transforms.Resize((300, 300))(dif)



        #dif = dif[1][0, :, t:300+t, t:300+t]
        dif = np.array(dif.repeat(3, 1, 1))

        dif_im = dif.copy()
#        dif_im = ((dif_im - dif_im.min())/(dif_im.max() - dif_im.min()))
        dif_im = (dif_im+1.5)/3

        off = 0.3

        center_width = 0.2
        overlap = 0.1

        left = 0.5 - center_width / 2 + overlap + off
        right = 0.5 + center_width/2 - overlap + off

        z_l = 0.5 - center_width/2 - overlap + off
        z_r = 0.5 + center_width/2 + overlap + off

        dif_im[0] = 1-np.minimum(dif_im[0], left)/left
        dif_im[1] = np.maximum(dif_im[1]-right, 0)/(1-right)
        dif_im[2] = np.maximum(np.minimum(dif_im[2], z_r)-z_l, 0)/(center_width+2*overlap)

        dif_im[2] = np.sin(np.pi*dif_im[2]) #-(dif_im[2]*(dif_im[2]-1))
        #dif_im[2] = 0
       # dif_im[0,:,:] = np.maximum(-dif_im[0,:,:], 0)
       # dif_im[1,:,:] = np.maximum(dif_im[1,:,:], 0)

       # c = dif_im[2,:,:]
       # dif_im[2,:,:] = 0.1# (c-c.min())/(c.max()-c.min())

        #dif = (dif - dif.min()) / (dif.max() - dif.min())
       # threshold = 0.2
        threshold = filters.threshold_otsu(dif)

        dif_idx = dif < threshold
        dif = dif_idx.astype(np.float)


        combined = copy.deepcopy(input)
        combined[~dif_idx] = 0

        av = copy.deepcopy(input)
        av[~dif_idx] = np.mean(av)

        mask = np.zeros((3, 300, 300))
#        mask[0,dif_idx[0,:,:]] = 0.3
#        mark = copy.deepcopy(input) + mask
#        mark = (mark - mark.min()) / (mark.max() - mark.min())
        mark = copy.deepcopy(input)
        mark[0, dif_idx[0, :, :]] = 1


        self.ax_image.clear()
        self.ax_image.imshow(input.transpose((1, 2, 0)))
        self.ax_pp_image.clear()
        self.ax_pp_image.imshow(output.transpose((1, 2, 0)))
        self.ax_dif_image.clear()
        self.ax_dif_image.imshow(dif.transpose((1, 2, 0)))
        self.ax_comb_image.clear()
        self.ax_comb_image.imshow(combined.transpose((1, 2, 0)))
        self.ax_image_m.clear()
        self.ax_image_m.imshow(mark.transpose((1, 2, 0)))
        self.ax_av.clear()
        self.ax_av.imshow(av.transpose((1, 2, 0)))
        self.ax_dif_im.clear()
        self.ax_dif_im.imshow(dif_im.transpose((1, 2, 0)))
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
        self.ax_image = self.image_figure.add_subplot(4, 2, 1)
        self.ax_image_m = self.image_figure.add_subplot(4, 2, 5)
        self.ax_pp_image = self.image_figure.add_subplot(4, 2, 3)
        self.ax_dif_image = self.image_figure.add_subplot(4, 2, 2)
        self.ax_comb_image = self.image_figure.add_subplot(4, 2, 4)
        self.ax_av = self.image_figure.add_subplot(4, 2, 6)
        self.ax_dif_im = self.image_figure.add_subplot(4, 2, 7)

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
        label_layout, panel = add_hlayout(main_layout)
        panel.setFixedHeight(50)
        label_layout.addWidget(self.label_cracked)
        label_layout.addWidget(self.label_inactive)
        button_layout, panel = add_hlayout(main_layout)
        panel.setFixedHeight(50)
        button_layout.addWidget(button_prev)
        button_layout.addWidget(button_next)
        button_layout.addWidget(button_refresh)
       # main_layout.addWidget(self.loss_canvas)

        # initialization
        self.setCentralWidget(main_panel)
        main_panel.setLayout(main_layout)
        button_prev.clicked.connect(self.on_prev_image_click)
        button_next.clicked.connect(self.on_next_image_click)
        button_refresh.clicked.connect(self.on_refresh_image_click)
