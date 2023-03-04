from torch import nn


class UnetUpsampleBlock(nn.Module):

    def __init__(self, in_channel, skip_cnt, out_channel,
                 padding, output_padding):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_channel, out_channel,
                                        kernel_size=2, stride=2,
                                        padding=padding, output_padding=output_padding)
        self.bn1 = nn.BatchNorm2d(out_channel)
    def forward(self, x, skip):
