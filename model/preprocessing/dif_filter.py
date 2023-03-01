import numpy as np
import scipy
import torch
import torchvision.transforms.functional


class DiffusionFilter:

    def __init__(self, cuda=False):
        self.cuda = cuda

        self.ep = 0.2
        self.b = 0.0000001

    def calc_gradients(self, image):

        n = np.empty((4, image.shape[0], image.shape[1]))
        t = np.concatenate((image[:1, :], image[:-1, :]))
        n[0, ...] = np.concatenate((image[:1, :], image[:-1, :]))[:, :]
        n[1, ...] = np.concatenate((image[1:, :], image[-1:, :]))[:, :]
        n[2, ...] = np.concatenate((image[:, :1], image[:, :-1]), axis=1)[:, :]
        n[3, ...] = np.concatenate((image[:, 1:], image[:, -1:]), axis=1)[:, :]

        image = np.repeat(image[np.newaxis], 4, axis=0)

        grad = image - n
        return grad

    def threshold(self, image):
        return 1/(1+np.exp(-self.b * image - self.ep))

    def calc_coef(self, g, grad):
        return 1 - np.power(1 + (grad*grad)/(g*g), -1)

    def perform_iteration(self, image, g):
        grad = self.calc_gradients(image)

        c = np.empty((4, image.shape[0], image.shape[1]))
        for i in range(4):
            c[i] = self.calc_coef(g, grad[i])

        return np.sum(c*grad, axis=0)/4

    def filter(self, image):
        image = np.array(image[0, ...])

        g = self.threshold(image)

        for i in range(1):
            image = self.perform_iteration(image, g)

        image = np.repeat(image[np.newaxis, ...], 3, axis=0)

        return torch.tensor(image, dtype=torch.float)
