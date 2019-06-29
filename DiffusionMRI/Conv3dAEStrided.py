import torch.nn as nn
from torch.nn.functional import interpolate


class ConvEncoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encode = nn.Sequential(
            # N x 7 x H x W x D --> N x 7 x H/2 x W/2 x D/2
            nn.Conv3d(
                in_channels=7,
                out_channels=7,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
        )
        self.input_size = input_size

    def forward(self, x):
        out = self.encode(x)
        out = interpolate(out, size=self.input_size, mode='trilinear', align_corners=True)
        return out


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(
            # N x 7 x H/2 x W/2 x D/2 --> N x 7 x H x W x D
            nn.Conv3d(
                in_channels=7,
                out_channels=7,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
        )

    def forward(self, h):
        y = self.decode(h)
        return y
