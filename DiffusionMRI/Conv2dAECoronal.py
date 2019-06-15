import torch.nn as nn
from torch.nn.functional import interpolate


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = nn.Sequential(
            # N x 288 x 145 x 145 --> N x 4 x 72 x 72
            nn.Conv2d(
                in_channels=288,
                out_channels=4,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(
            # N x 4 x 72 x 72 --> N x 288 x 145 x 145
            nn.ConvTranspose2d(
                in_channels=4,
                out_channels=288,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            )
        )

    def forward(self, h):
        y = self.decode(h)
        return y
