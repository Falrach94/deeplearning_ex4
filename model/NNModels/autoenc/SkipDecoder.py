import torch
import torch.nn as nn


train_mean = 0.59685254
train_std = 0.16043035

class UpsampleBlock(torch.nn.Module):
    def __init__(self, in_channels, skip_cnt, out_channels,
                 padding=0, out_padding=0, stride=2):
        super().__init__()

        self.skip_con = nn.Conv2d(skip_cnt, out_channels, kernel_size=1)

        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels,
                                             kernel_size=3, stride=stride,
                                             padding=padding, output_padding=out_padding)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.drop = nn.Dropout(p=0.99)

    def forward(self, x, skip):

        x = self.trans_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        skip = self.skip_con(skip)
        skip = self.bn2(skip)
        skip = self.relu(skip)
        skip = self.drop(skip)
#        skip = nn.functional.dropout(skip, p=0.99)

        if skip is not None:
            x = torch.concat((x, skip), dim=1)

        return x


class ResBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.ds = in_channels != out_channels
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 1),
                                        nn.BatchNorm2d(out_channels))
    def forward(self, x):
        res = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.ds:
            x = x + self.downsample(res)
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

        skip_sizes = [256, 128, 64, 64]

        self.upsample1 = UpsampleBlock(in_channels=in_channel, skip_cnt=skip_sizes[0],
                                       out_channels=256, padding=1)  # (512+0)x10x10 -> 256x19x19
        self.res1 = ResBlock(skip_sizes[0]+256, 256)

        self.upsample2 = UpsampleBlock(in_channels=256, skip_cnt=skip_sizes[1],
                                       out_channels=128,
                                       padding=1, out_padding=1)  # (256+)x19x19 -> 38x38
        self.res2 = ResBlock(skip_sizes[1]+128, 128)

        self.upsample3 = UpsampleBlock(in_channels=128, skip_cnt=skip_sizes[2],
                                       out_channels=64,
                                       padding=1)  # 10x10 -> 75x75
        self.res3 = ResBlock(skip_sizes[2]+64, 64)

        self.upsample4 = UpsampleBlock(in_channels=64, skip_cnt=skip_sizes[3],
                                       out_channels=64,
                                       padding=1, out_padding=1)  # 10x10 -> 150x150
        self.res4 = ResBlock(skip_sizes[3]+64, 32)

        self.upsample = nn.Sequential(
            nn.Upsample((300, 300)),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, skip):
        x = self.upsample1(x, skip[3])
        x = self.res1(x)
        x = self.upsample2(x, skip[2])
        x = self.res2(x)
        x = self.upsample3(x, skip[1])
        x = self.res3(x)
        x = self.upsample4(x, skip[0])
        x = self.res4(x)

        x = self.upsample(x)
        x = self.sigmoid(x)
        x = (x-train_mean)/train_std
        return x

