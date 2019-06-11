import torch.nn as nn
from torch.nn.functional import interpolate


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.encode = nn.Sequential(
            # N x C_in x H x W --> N x C_out x H x W
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

        # pool by factor of 2
        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.encode(x)
        out = self.max_pool(out)
        return out


class ConvDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, out_size):
        super().__init__()

        self.out_size = out_size

        self.decode = nn.Sequential(
            #  N x 72 x 145 x 145 --> N x 288 x 145 x 145
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = interpolate(x, size=self.out_size)
        out = self.decode(out)
        return out
