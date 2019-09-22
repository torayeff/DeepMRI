import torch.nn as nn
from torch.nn.functional import interpolate


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(
                in_channels=288,
                out_channels=144,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(144),

            nn.Conv2d(
                in_channels=144,
                out_channels=72,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(72),

            nn.Conv2d(
                in_channels=72,
                out_channels=36,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(36)
        )

    def forward(self, x, return_all=False):
        out = self.encode(x)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode1 = nn.Sequential(
            nn.Conv2d(
                in_channels=36,
                out_channels=72,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.PReLU(72)
        )

        self.decode2 = nn.Sequential(
            nn.Conv2d(
                in_channels=72,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.PReLU(144)
        )

        self.decode3 = nn.Sequential(
            nn.Conv2d(
                in_channels=144,
                out_channels=288,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            )
        )

    def forward(self, h):
        y = interpolate(h, size=(35, 35), mode='bilinear', align_corners=True)
        y = self.decode1(y)

        y = interpolate(y, size=(72, 72), mode='bilinear', align_corners=True)
        y = self.decode2(y)

        y = interpolate(y, size=(145, 145), mode='bilinear', align_corners=True)
        y = self.decode3(y)
        return y
