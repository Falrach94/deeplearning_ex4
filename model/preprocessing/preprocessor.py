import torch
from skimage.color import gray2rgb
import numpy as np
import scipy.fft
import torchvision as tv

from model.preprocessing.dif_filter import DiffusionFilter
from model.preprocessing.freq_filter import FrequencyFilter

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class Preprocessor:

    def __init__(self):
        self.freq_fiter = FrequencyFilter(w=6, d=10, sig=12)

        self.transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                               tv.transforms.ToTensor(),
                               tv.transforms.Normalize(train_mean, train_std)])

        self.dif_filter = DiffusionFilter()

    def process(self, image):
        image = gray2rgb(image)
        image = self.transform(image)

        image = self.freq_fiter.filter(image)
      #  image = self.dif_filter.filter(image)

        return image

