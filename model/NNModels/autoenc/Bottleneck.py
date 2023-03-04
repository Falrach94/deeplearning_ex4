from torch import nn

### 512x10x10 -> 512x10x10
class Bottleneck(nn.Module):

    def __init__(self, bottleneck_size, bottleneck_activation):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, bottleneck_size)
        )

        self.sig = bottleneck_activation

        self.upsample = nn.Sequential(
            nn.Linear(bottleneck_size, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (512, 1, 1)),
            nn.Conv2d(512, 256*10*10, kernel_size=1),
            nn.Flatten(),
            nn.Unflatten(1, (256, 10, 10)),
            nn.BatchNorm2d(256)
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.sig(x)
        x = self.upsample(x)
        return x

