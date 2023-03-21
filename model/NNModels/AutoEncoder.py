import torch
from torch import nn
from torch.nn import init

from cli_program.settings.behaviour_settings import BEST_MODEL_PATH


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

    def __init__(self):
        super().__init__()

        self.initial_conv = nn.Sequential(nn.Conv2d(3, 64, 7, 2),  # 300x300 -> 293x293 ->146x146 -> 147x147
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d(kernel_size=3, stride=2))  # 147x147->144x144 -> 72x72 -> 73x73

        self.l1 = self._make_layer(64, 64, 3)  # 73x73 -> 73x73
        self.l2 = self._make_layer(64, 128, 4)  # 71x71 -> 35x35
        self.l3 = self._make_layer(128, 256, 6)  # 35x35 -> 17x17
        self.l4 = self._make_layer(256, 512, 3)  # 17x17 -> 10x10

        # ResNet34
#        layers = [
#            self.l1,
#            self.l2,
#            self.l3,
#            self.l4
#        ]
       # self.feature_extraction = nn.Sequential(*layers)

 #       self.output_layer = nn.Sequential(
 #       )

    def forward(self, x):
        x = self.initial_conv(x)
        skip1 = self.l1(x)
        skip2 = self.l2(skip1)
        skip3 = self.l3(skip2)
        skip4 = self.l4(skip3)
        return skip4, skip3

        #x = self.feature_extraction(x)
#        x = self.output_layer(x)
        #return x

    @staticmethod
    def _make_layer(in_channels, out_channels, reps):
        cut_in_half = in_channels != out_channels
        layers = [ResNetBlock(in_channels, out_channels, stride=(2 if cut_in_half else 1))]
        layers += [ResNetBlock(out_channels, out_channels, 1)]*(reps-1)
        return nn.Sequential(*layers)


class Bottleneck(nn.Module):

    def __init__(self):
        super().__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 8, kernel_size=1, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

     #   self.bottleneck = nn.Sequential(
     #       nn.AdaptiveAvgPool2d((1,1)),
     #       nn.Flatten(),
     #       nn.Linear(512, 128 * 9 * 9),
     #       nn.ReLU(inplace=True),
     #       nn.Dropout(p=0.5)
     #   )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)

        self.last_activation = None

      #  layers = [
      #      nn.AdaptiveAvgPool2d((1, 1)),
      #      nn.Flatten(),
      #  ]
      #  self.output_layer = nn.Sequential(*layers)

      #  self.bottleneck = nn.Sequential(
      #      nn.Linear(512, sparse_cnt),
      #      nn.ReLU(inplace=True),
      #  )
      #  self.expander = nn.Sequential(
      #      nn.Linear(sparse_cnt, 512),
      #      nn.ReLU(inplace=True)
      #  )

        self._sparse_output = False


    def forward(self, x):

       # x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        #x = x.view(x.size(0), 256, 9, 9)

        #self.last_activation = x

        #if not self._sparse_output:
        #    x = self.expander(x)
        #return x, self.last_activation
        return x

class TestDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential()

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
        #x = self.fc(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.sig(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, padding, out_padding, fct=nn.ReLU(inplace=True)):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=padding,
                               output_padding=out_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if skip_channels == 0:
            self.skip = None
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(skip_channels, skip_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(skip_channels),
                nn.ReLU(inplace=True)
            )

        self.filter = nn.Sequential(
            nn.Conv2d(out_channels+skip_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            fct
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            skip = self.skip(skip)
            x = torch.concat((x, skip), dim=1)
        x = self.filter(x)
        return x

class Decoder2(nn.Module):

    def __init__(self):
        super().__init__()

        self.l1 = UpsampleBlock(512, 256, 0, 1, 0)  # 10 -> 19
        self.l2 = UpsampleBlock(256, 128, 0, 1, 0)  # 19 -> 37
        self.l3 = UpsampleBlock(128, 64, 0, 0, 0)  # 37->75
        self.l4 = UpsampleBlock(64, 32, 0, 1, 1)  # 75->150
        self.l5 = UpsampleBlock(32, 1, 0, 1, 1, nn.Sigmoid())  # 150-300


    def forward(self, x, skip):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        return x

class ResNetAutoEncoder(torch.nn.Module):


    def __init__(self, load=False):
        super().__init__()

        '''
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), #150
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )'''
        '''
        self.bottleneck = nn.Sequential(
            nn.Linear(75*75*16, 500),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(500, 75*75*16),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True)
        )'''
        self.bottleneck = Bottleneck()

        self.decoder = Decoder2()

        self.encoder = Encoder()

       # print('encoder', sum(p.numel() for p in self.encoder.parameters()))
       # print('bottleneck', sum(p.numel() for p in self.bottleneck.parameters()))
       # print('decoder', sum(p.numel() for p in self.decoder.parameters()))

        #self.encoder = Encoder(sparse_cnt)
        #self.bottleneck = Bottleneck(sparse_cnt)

        #self.decoder = TestDecoder()
        #self.classifier = nn.Sequential(nn.Linear(sparse_cnt, 2),
        #                                nn.Sigmoid())
        #self._sparse_output = False

        if load:
            state = torch.load(BEST_MODEL_PATH)
            self.load_state_dict(state)

    def forward(self, x):
        x, skip = self.encoder(x)
        #x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
#        x = x.view(x.size(0), 128, 9, 9)
        x = self.decoder(x, skip)

        #x = x.view(x.size(0), 300, 300)


        mean = 0.59685254
        std = 0.16043035

        return (x-mean)/std#, y_s, x_s
