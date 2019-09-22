import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encode_local = nn.Sequential(
            nn.Conv2d(
                in_channels=288,
                out_channels=8,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.PReLU(8)
        )

        self.encode_regional_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=288,
                out_channels=8,
                kernel_size=5,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(8)
        )

        self.encode_regional_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=288,
                out_channels=16,
                kernel_size=5,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(16),

            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=5,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(8)
        )

        self.encode_regional_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=288,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(32),

            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=5,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(16),

            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=5,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(8)
        )

        self.input_size = input_size

    def forward(self, x):
        out_local = self.encode_local(x)

        out_regional_1 = self.encode_regional_1(x)
        out1 = interpolate(out_regional_1, size=self.input_size, mode='bilinear', align_corners=True)

        out_regional_2 = self.encode_regional_2(x)
        out2 = interpolate(out_regional_2, size=self.input_size, mode='bilinear', align_corners=True)

        out_regional_3 = self.encode_regional_3(x)
        out3 = interpolate(out_regional_3, size=self.input_size, mode='bilinear', align_corners=True)

        out = torch.cat([out_local, out1, out2, out3], dim=1)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.PReLU(64),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.PReLU(128),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.PReLU(256),

            nn.Conv2d(
                in_channels=256,
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
