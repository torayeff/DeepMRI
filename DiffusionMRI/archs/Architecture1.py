import torch.nn as nn
from torch.nn.functional import interpolate


class ConvEncoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encode1 = nn.Sequential(
            nn.Conv2d(
                in_channels=288,
                out_channels=7,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(7),
        )
        self.encode2 = nn.Sequential(
            nn.Conv2d(
                in_channels=7,
                out_channels=7,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(7),
        )

        self.input_size = input_size

    def forward(self, x, return_all=False):
        out1 = self.encode1(x)
        out2 = self.encode2(out1)
        out = interpolate(out2, size=self.input_size, mode='bilinear', align_corners=True)
        if return_all:
            return out1, out2, out
        return out


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(
            nn.Conv2d(
                in_channels=7,
                out_channels=288,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            )
        )

    def forward(self, h):
        y = self.decode(h)
        return y
