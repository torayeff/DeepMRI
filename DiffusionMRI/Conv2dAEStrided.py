import torch.nn as nn
from torch.nn.functional import interpolate


class ConvEncoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encode = nn.Sequential(
            # N x 288 x H x W --> N x 72 x H/2 x W/2
            nn.Conv2d(
                in_channels=288,
                out_channels=72,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(72),

            # N x 72 x H/2 x W/2 --> N x 36 x H/4 x W/4
            nn.Conv2d(
                in_channels=72,
                out_channels=36,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(36),
        )

        self.input_size = input_size

    def forward(self, x):
        out = self.encode(x)
        out = interpolate(out, size=self.input_size, mode='bilinear', align_corners=True)
        return out


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(
            # N x 36 x H x W --> N x 288 x H x W
            nn.Conv2d(
                in_channels=36,
                out_channels=288,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
        )

    def forward(self, h):
        y = self.decode(h)
        return y
