from torch import nn

from model.NNModels.autoenc.Bottleneck import Bottleneck
from model.NNModels.autoenc.SkipDecoder import SkipDecoder
from model.NNModels.autoenc.SkipEncoder import SkipEncoder


class SkipAutoEncoder(nn.Module):
    def __init__(self, bottleneck_size=128, bottleneck_activation=nn.Sigmoid()):
        super().__init__()
        self.encoder = SkipEncoder()
        self.decoder = SkipDecoder(512)
        self.bottleneck = Bottleneck(bottleneck_size, bottleneck_activation)

    def forward(self, x):
        x, skip = self.encoder(x)
        #x = self.bottleneck(x)
        x = self.decoder(x, skip)
        return x

