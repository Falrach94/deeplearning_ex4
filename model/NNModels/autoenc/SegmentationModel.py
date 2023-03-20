import numpy as np
import torch
import torchmetrics
from skimage import filters
from torch import nn

from model.NNModels.AutoEncoder import ResNetAutoEncoder


class SegmentationModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = ResNetAutoEncoder(256, load=True)
        kernel_size = 9
        self.ssim = torchmetrics.StructuralSimilarityIndexMeasure(gaussian_kernel=False,
                                                                  kernel_size=(kernel_size, kernel_size),
                                                                  reduction='none',
                                                                  return_full_image=True)
        self.cut = 1

    def forward(self, x):
        identity = x
        clean = self.model(x)

        MEAN = 0.59685254
        STD = 0.16043035
        MAX = MEAN / STD

        image_tensor = torch.mean(x, dim=1, keepdim=True)
        clean_tensor = torch.mean(clean, dim=1, keepdim=True)

        image_tensor = image_tensor.cpu().detach()
        clean_tensor = clean_tensor.cpu().detach()
        dif = self.ssim(image_tensor, clean_tensor)[1]

        dif = dif[:, :, self.cut:300+self.cut, self.cut:300+self.cut]
        thresholds = torch.tensor([filters.threshold_otsu(np.array(d)) for i, d in enumerate(dif)])
        thresholds = thresholds[:, None, None, None].repeat(1, 1, 300, 300)
        dif_idx = dif < thresholds
        dif = (dif_idx.float() - MEAN)/STD
        identity[:, 0] = dif[:, 0]
        #dif = dif_idx.astype(np.float)

        return identity

