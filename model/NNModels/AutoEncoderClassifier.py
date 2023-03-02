from torch import nn

from model.NNModels.AutoEncoder import Encoder


class ResNet34AutoEnc(nn.Module):

    def __init__(self, autoencoder=None):
        super().__init__()
        if autoencoder is not None:
            self.encoder = autoencoder.encoder
        else:
            self.encoder = Encoder()
        self.classifier = nn.Sequential(
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

