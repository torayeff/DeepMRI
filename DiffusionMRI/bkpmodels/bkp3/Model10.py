import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encode_local = nn.Sequential(
            nn.Conv2d(
                in_channels=288,
                out_channels=88,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.PReLU(88),

            nn.Conv2d(
                in_channels=88,
                out_channels=44,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.PReLU(44),

            nn.Conv2d(
                in_channels=44,
                out_channels=22,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.PReLU(22)
        )

        self.encode_regional = nn.Sequential(
            nn.Conv2d(
                in_channels=288,
                out_channels=88,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=True
            ),
            nn.PReLU(88),

            nn.Conv2d(
                in_channels=88,
                out_channels=44,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=True
            ),
            nn.PReLU(44),

            nn.Conv2d(
                in_channels=44,
                out_channels=22,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=True
            ),
            nn.PReLU(22)
        )

        self.input_size = input_size

    def forward(self, x):
        out1 = self.encode_local(x)
        out2 = self.encode_regional(x)
        out3 = interpolate(out2, size=self.input_size, mode='bilinear', align_corners=True)

        out = torch.cat([out1, out3], dim=1)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(

            nn.Conv2d(
                in_channels=44,
                out_channels=88,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.PReLU(88),

            nn.Conv2d(
                in_channels=88,
                out_channels=176,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.PReLU(176),

            nn.Conv2d(
                in_channels=176,
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
