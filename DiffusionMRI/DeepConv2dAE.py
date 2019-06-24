import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = nn.Sequential(
            # N x 288 x H x W --> N x 144 x H x W
            nn.Conv2d(
                in_channels=288,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(144),

            # N x 144 x H x W --> N x 72 x H x W
            nn.Conv2d(
                in_channels=144,
                out_channels=72,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(72),

            # N x 72 x H x W --> N x 36 x H x W
            nn.Conv2d(
                in_channels=72,
                out_channels=36,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(36),

            # N x 36 x H x W --> N x 18 x H x W
            nn.Conv2d(
                in_channels=36,
                out_channels=18,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(18),

            # N x 18 x H x W --> N x 7 x H x W
            nn.Conv2d(
                in_channels=18,
                out_channels=7,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(7),
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(
            # N x 36 x H x W --> N x 288 x H x W
            nn.Conv2d(
                in_channels=7,
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
