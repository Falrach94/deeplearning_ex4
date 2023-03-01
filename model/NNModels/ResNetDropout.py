import torch
import torch.nn as nn


class ResNetDropout(torch.nn.Module):

    def __init__(self, dropout):
        super().__init__()

        self.dropout = dropout

        self.conv = torch.nn.Conv2d(3, 64, 7, 2)
        self.norm = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(3, 2)
        self.res_block1 = ResBlock(64, 64, 1, dropout)
        self.res_block2 = ResBlock(64, 128, 2, dropout)
        self.res_block3 = ResBlock(128, 256, 2, dropout)
        self.res_block4 = ResBlock(256, 512, 2, dropout)
        self.av_pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(512, 2)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        x = torch.nn.functional.dropout(x, training=self.training, inplace=False)

        x = self.pool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.av_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sig(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout):
        super().__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride),
                                        nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU()

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = torch.nn.functional.dropout(out, training=self.training, inplace=False)

        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.relu(out)

        if identity.size != out.size:
            identity = self.downsample(x)

        y = out + identity
        out = torch.relu(y)

        out = torch.nn.functional.dropout(out, training=self.training, inplace=False)

        return out

