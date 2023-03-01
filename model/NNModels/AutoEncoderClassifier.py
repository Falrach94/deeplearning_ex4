from torch import nn


class ResNet34AutoEnc(nn.Module):

    def __init__(self, autoencoder):
        super().__init__()
        self.encoder = autoencoder.encoder
        self.classifier = nn.Sequential(
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

