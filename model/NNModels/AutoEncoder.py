import torch
from torch import nn


class ResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride),
                                        nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU()

        self.stride = stride

        self._sparse_output = False


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if identity.size != out.size:
            identity = self.downsample(x)

        y = out + identity
        out = self.relu(y)

        return out


class Encoder(nn.Module):

    def __init__(self, sparse_cnt):
        super().__init__()

        self.initial_conv = nn.Sequential(nn.Conv2d(3, 64, 7, 2),  # 300x300 -> 293x293 ->146x146 -> 147x147
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d(kernel_size=3, stride=2))  # 147x147->144x144 -> 72x72 -> 73x73

        # ResNet34
        layers = [
            self._make_layer(64, 64, 3),  # 71x71 -> 71x71
            self._make_layer(64, 128, 4),  # 71x71 -> 35x35
            self._make_layer(128, 256, 6),  # 35x35 -> 17x17
            self._make_layer(256, 512, 3)  # 17x17 -> 10x10
        ]
        self.feature_extraction = nn.Sequential(*layers)

        layers = [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ]
        self.output_layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.feature_extraction(x)
        x = self.output_layer(x)
        return x

    @staticmethod
    def _make_layer(in_channels, out_channels, reps):
        cut_in_half = in_channels != out_channels
        layers = [ResNetBlock(in_channels, out_channels, stride=(2 if cut_in_half else 1))]
        layers += [ResNetBlock(out_channels, out_channels, 1)]*(reps-1)
        return nn.Sequential(*layers)


class Bottleneck(nn.Module):

    def __init__(self, sparse_cnt):
        super().__init__()

        self.last_activation = None

        self.bottleneck = nn.Sequential(
            nn.Linear(512, sparse_cnt),
            nn.ReLU(inplace=True),
        )
        self.expander = nn.Sequential(
            nn.Linear(sparse_cnt, 512),
            nn.ReLU(inplace=True)
        )

        self._sparse_output = False

    def sparse(self):
        self._sparse_output = True

    def autoencode(self):
        self._sparse_output = False

    def forward(self, x):
        x = self.bottleneck(x)

        self.last_activation = x

        if not self._sparse_output:
            x = self.expander(x)
        return x, self.last_activation

class TestDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(512, 256*9*9)

        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1, stride=1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 9, 9)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.sig(x)
        return x


class SimpleDecoder(nn.Module):
    def __init__(self):
        super(SimpleDecoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # reshape from a flattened tensor to a 4-dimensional tensor
        x = x.view(x.size(0), 512, 1, 1)

        x = self.conv1(self.upsample(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.upsample(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.upsample(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.upsample(x))
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(self.upsample(x))
        x = self.sigmoid(x)

        return x


class ResNetAutoEncoder(torch.nn.Module):

    def __init__(self, sparse_cnt=128):
        super().__init__()

        self.encoder = Encoder(sparse_cnt)
        self.bottleneck = Bottleneck(sparse_cnt)
        self.decoder = TestDecoder()
        self.classifier = nn.Sequential(nn.Linear(sparse_cnt, 2),
                                        nn.Sigmoid())
        self._sparse_output = False

    def sparse(self):
        self._sparse_output = True
        self.bottleneck.sparse()

    def autoencode(self):
        self._sparse_output = False
        self.bottleneck.autoencode()

    def forward(self, x):
        x = self.encoder(x)
        x, x_s = self.bottleneck(x)
#        if not self._sparse_output:

        x = self.decoder(x)
       # y_s = self.classifier(x_s)

        return x#, y_s, x_s
