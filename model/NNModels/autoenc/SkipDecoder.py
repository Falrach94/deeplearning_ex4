import torch
import torch.nn as nn


train_mean = 0.59685254
train_std = 0.16043035


class DecoderBlock(torch.nn.Module):

    def __init__(self, in_channels, skip_channels,
                 out_channels,
                 padding=0, out_padding=0, stride=2):
        super().__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels,
                                             kernel_size=3, stride=stride,
                                             padding=padding, output_padding=out_padding)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.trans_conv(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = torch.concat((x, skip), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x



'''
Expects input sizes of
 - 10x10
 - 
'''
class SkipDecoder(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        self.dec1 = DecoderBlock(in_channel, 256, 256, padding=1)  # 10x10 -> 19x19
        self.dec2 = DecoderBlock(256, 128, 128, padding=1, out_padding=1)  # 19x19 -> 38x38
        self.dec3 = DecoderBlock(128, 64, 64, padding=1)  # 38x38 -> 75x75
        self.dec4 = DecoderBlock(64, 64, 64, padding=1, stride=1)  # 75x75 -> 75x73
        self.dec5 = DecoderBlock(64, 64, 32, padding=1, out_padding=1)  # 75x75 -> 150x150
        self.upsample = nn.Sequential(
            nn.Upsample((300, 300)),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, skip):
        x = self.dec1(x, skip[4])
        x = self.dec2(x, skip[3])
        x = self.dec3(x, skip[2])
        x = self.dec4(x, skip[1])
        x = self.dec5(x, skip[0])
        x = self.upsample(x)
        x = self.sigmoid(x)
        x = (x-train_mean)/train_std
        return x
