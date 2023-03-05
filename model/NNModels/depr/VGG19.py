import torch


class Block(torch.nn.Module):


    def __init__(self, in_channels, out_channels, twice):
        super().__init__()
        if twice:
            self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                              torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                              torch.nn.Conv2d(out_channels, out_channels, 3, padding=1))
        else:
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=(1, 1))

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.pool = torch.nn.MaxPool2d((2, 2), stride=2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class VGG19(torch.nn.Module):

    def create_conv_layer(self, in_channels, out_channels):
        return [torch.nn.Conv2d(in_channels, out_channels,
                                kernel_size=(3, 3),
                                stride=(1, 1),
                                padding=(1, 1)),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True)]

    def __init__(self):
        super().__init__()

        layers = []
        layers += self.create_conv_layer(3, 64)
        layers += self.create_conv_layer(64, 64)
        layers += [torch.nn.MaxPool2d((2, 2), stride=(2, 2))]
        layers += self.create_conv_layer(64, 128)
        layers += self.create_conv_layer(128, 128)
        layers += [torch.nn.MaxPool2d((2, 2), stride=(2, 2))]
        layers += self.create_conv_layer(128, 256)
        layers += self.create_conv_layer(256, 256)
        layers += self.create_conv_layer(256, 256)
        layers += [torch.nn.MaxPool2d((2, 2), stride=(2, 2))]
        layers += self.create_conv_layer(256, 512)
        layers += self.create_conv_layer(512, 512)
        layers += self.create_conv_layer(512, 512)
        layers += [torch.nn.MaxPool2d((2, 2), stride=(2, 2))]
        layers += self.create_conv_layer(512, 512)
        layers += self.create_conv_layer(512, 512)
        layers += self.create_conv_layer(512, 512)
        layers += [torch.nn.MaxPool2d((2, 2), stride=(2, 2))]

        self.conv = torch.nn.Sequential(*layers)

        layers = []
        layers += [torch.nn.Linear(7*7*512, 4096),
                   torch.nn.ReLU(inplace=True),
                   torch.nn.Dropout()]
        layers += [torch.nn.Linear(4096, 4096),
                   torch.nn.ReLU(inplace=True),
                   torch.nn.Dropout()]
        layers += [torch.nn.Linear(4096, 2),
                   torch.nn.ReLU(inplace=True),
                   torch.nn.Dropout()]

        self.fc = torch.nn.Sequential(*layers)

        self.flatten = torch.nn.Flatten()

        self.sigmoid = torch.nn.Sigmoid()

        '''self.block1 = Block(3, 64, False)
        self.block2 = Block(64, 128, False)
        self.block3 = Block(128, 256, True)
        self.block4 = Block(256, 512, True)
        self.block5 = Block(512, 512, True)
        self.fc1 = torch.nn.Linear(512*49, 4096)
        self.fc2 = torch.nn.Linear(4096, 2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.flatten = torch.nn.Flatten()'''

    def forward(self, x):

        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.sigmoid(x)

        '''
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        return self.sigmoid(x)
        '''

