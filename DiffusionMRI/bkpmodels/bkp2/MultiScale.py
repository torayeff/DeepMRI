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
            nn.PReLU(22),

            nn.Conv2d(
                in_channels=22,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.PReLU(3)
        )

        self.encode_regional_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=288,
                out_channels=88,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(88),

            nn.Conv2d(
                in_channels=88,
                out_channels=44,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.PReLU(44),

            nn.Conv2d(
                in_channels=44,
                out_channels=22,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.PReLU(22),

            nn.Conv2d(
                in_channels=22,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.PReLU(3)
        )

        self.encode_regional_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=288,
                out_channels=88,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(88),

            nn.Conv2d(
                in_channels=88,
                out_channels=44,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(44),

            nn.Conv2d(
                in_channels=44,
                out_channels=22,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.PReLU(22),

            nn.Conv2d(
                in_channels=22,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.PReLU(3)
        )

        self.encode_regional_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=288,
                out_channels=88,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(88),

            nn.Conv2d(
                in_channels=88,
                out_channels=44,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(44),

            nn.Conv2d(
                in_channels=44,
                out_channels=22,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(22),

            nn.Conv2d(
                in_channels=22,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.PReLU(3)
        )

        self.encode_regional_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=288,
                out_channels=88,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(88),

            nn.Conv2d(
                in_channels=88,
                out_channels=44,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(44),

            nn.Conv2d(
                in_channels=44,
                out_channels=22,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(22),

            nn.Conv2d(
                in_channels=22,
                out_channels=3,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(3)
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

        out_regional_4 = self.encode_regional_4(x)
        out4 = interpolate(out_regional_4, size=self.input_size, mode='bilinear', align_corners=True)

        out = torch.cat([out_local, out1, out2, out3, out4], dim=1)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(
            nn.Conv2d(
                in_channels=15,
                out_channels=44,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.PReLU(44),

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
