from torch import nn


class ConvAutoEncoder(nn.Model):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(90000, 500),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(500, 90000)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1)
        )