import numpy as np
import scipy
import torch
import torchvision.transforms.functional


class FrequencyFilter:

    def __init__(self, w, d, sig, cuda=False):
        self.w = w
        self.d = d
        self.sig = sig
        self.cuda = cuda
        self.recalc_filter()

    def D(self, uv, mn):
        return np.sqrt(np.power(uv[0]-mn[0]/2, 2), np.power(uv[1]-mn[1]/2, 2))

    def recalc_filter(self):
        WIDTH = 300
        HEIGHT = 300
        self.filter_mat = self.calc_filter((WIDTH, HEIGHT)).repeat((3,1,1))
        if self.cuda:
            self.filter_mat = self.filter_mat.cuda()

    def calc_filter(self, size):
        grid = np.meshgrid(range(0, size[0]), range(0, size[1]))
        grid = np.stack((grid[0], grid[1]))

        D = self.D(grid, size)

        zeros_a = D >= self.d
        zeros_b = size[1]/2-self.w <= D
        zeros_c = D <= size[1]/2+self.w

        zeros = np.logical_and(np.logical_and(zeros_a, zeros_b), zeros_c)
        print(self.sig)
        V = 1 - np.exp(-np.power(D, 2)/(2*self.sig**2))
        V[zeros] = 0
        return torch.tensor(V, dtype=torch.float)

    def filter(self, image):

        if image.is_cuda and not self.filter_mat.is_cuda:
            self.filter_mat = self.filter_mat.cuda()
        if not image.is_cuda and self.filter_mat.is_cuda:
            self.filter_mat = self.filter_mat.cpu()

        image = torch.fft.fft2(image)

        if len(image.shape) == 3:
            v = self.filter_mat
        else:
            v = self.filter_mat.repeat((image.shape[0], 1, 1, 1))

        image = torch.multiply(image, v)

        image = torch.fft.ifft2(image)

        image = torch.abs(image)

        return image
