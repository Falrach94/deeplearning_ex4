import torch
from torch import nn

from model.NNModels.autoenc.Bottleneck import Bottleneck
from model.NNModels.autoenc.SkipAutoEncoder import SkipAutoEncoder
from model.NNModels.autoenc.SkipDecoder import SkipDecoder
from model.NNModels.autoenc.SkipEncoder import SkipEncoder


class ScrambledAutoEncoder(SkipAutoEncoder):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, skip = self.encoder(x)
        #x = self.bottleneck(x)
        #x = torch.rand((1, 512, 10, 10))

        x = self.decoder(x, skip)
        return x

