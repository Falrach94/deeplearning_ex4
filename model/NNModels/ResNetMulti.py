import torch
import torch.nn as nn

from model.NNModels.ResNet import ResNet
from model.NNModels.ResNet50_pre import ResNet50_Pretrained
from model.NNModels.ResNetExtern import ResNet18Ex, ResNet152Ex, ResNet101Ex, ResNet50Ex, ResNet34Ex


class ResNetMulti(torch.nn.Module):

    def __init__(self, version):
        super().__init__()

        if version == 0:
            self.res1 = ResNet()
            self.res2 = ResNet()
            self.res3 = ResNet()
            self.res4 = ResNet()
        if version == 1:
            self.res1 = ResNet50_Pretrained()
            self.res2 = ResNet50_Pretrained()
            self.res3 = ResNet50_Pretrained()
            self.res4 = ResNet50_Pretrained()
        if version == 2:
            self.res1 = ResNet18Ex()
            self.res2 = ResNet18Ex()
            self.res3 = ResNet18Ex()
            self.res4 = ResNet18Ex()
        if version == 3:
            self.res1 = ResNet34Ex()
            self.res2 = ResNet34Ex()
            self.res3 = ResNet34Ex()
            self.res4 = ResNet34Ex()
        if version == 4:
            self.res1 = ResNet50Ex()
            self.res2 = ResNet50Ex()
            self.res3 = ResNet50Ex()
            self.res4 = ResNet50Ex()
        if version == 5:
            self.res1 = ResNet101Ex()
            self.res2 = ResNet101Ex()
            self.res3 = ResNet101Ex()
            self.res4 = ResNet101Ex()
        if version == 6:
            self.res1 = ResNet152Ex()
            self.res2 = ResNet152Ex()
            self.res3 = ResNet152Ex()
            self.res4 = ResNet152Ex()


    def forward(self, x):
#        y = torch.zeros([x.shape[0], 4, 2]).cuda()

        if self.training:

            y1 = self.res1(x[:, 0])[:, None, :]
            y2 = self.res2(x[:, 1])[:, None, :]
            y3 = self.res3(x[:, 2])[:, None, :]
            y4 = self.res4(x[:, 3])[:, None, :]

            return torch.cat((y1, y2, y3, y4), dim=1)
        else:
            y1 = self.res1(x)[None, ...]
            y2 = self.res2(x)[None, ...]
            y3 = self.res3(x)[None, ...]
            y4 = self.res4(x)[None, ...]
            return torch.mean(torch.cat((y1, y2, y3, y4)), dim=0)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
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

        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.relu(out)

        if identity.size != out.size:
            identity = self.downsample(x)

        y = out + identity
        out = torch.relu(y)

        return out

