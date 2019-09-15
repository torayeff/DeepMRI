import torch.nn as nn
from torch.nn.functional import interpolate


class Encoder(nn.Module):
    def __init__(self, input_size):
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
            nn.PReLU(36),

            nn.Conv2d(
                in_channels=36,
                out_channels=18,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(18),

            nn.Conv2d(
                in_channels=18,
                out_channels=9,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True
            ),
            nn.PReLU(9)
        )

        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=9,
                out_channels=9,
                kernel_size=50,
                stride=45,
                padding=0,
                output_padding=5
            )
        )

        self.input_size = input_size

    def forward(self, x, return_all=False):
        out = self.encode(x)
        # out = interpolate(out, size=self.input_size, mode='bilinear', align_corners=True)
        out = self.upconv(out)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(
            nn.Conv2d(
                in_channels=9,
                out_channels=18,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.PReLU(18),

            nn.Conv2d(
                in_channels=18,
                out_channels=36,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.PReLU(36),

            nn.Conv2d(
                in_channels=36,
                out_channels=72,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.PReLU(72),

            nn.Conv2d(
                in_channels=72,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.PReLU(144),

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
        y = self.decode(h)
        return y
