import torch
import torch.nn as nn


class DecoderBlock(torch.nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels, padding=0, out_padding=0):
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels,
                                             kernel_size=3, stride=2,
                                             padding=padding, out_padding=out_padding),
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels,
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

        x = torch.concat((x, skip), dim=2)

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
        self.dec1 = DecoderBlock(in_channel, 512, 256, padding=1)  # 10x10 -> 19x19
        self.dec2 = DecoderBlock(256, 256, 128, padding=1)  # 19x19 -> 37x37
        self.dec3 = DecoderBlock(128, 128, 64, padding=1)  # 37x37 -> 73x73
        self.dec4 = DecoderBlock(64, 64, 64)  # 73x73 -> 73x73
