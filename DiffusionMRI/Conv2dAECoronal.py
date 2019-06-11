import torch.nn as nn
from torch.nn.functional import interpolate


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = nn.Sequential(
            # N x 288 x 145 x 145 --> N x 144 x 145 x 145
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

            # N x 144 x 145 x 145 --> N x 144 x 72 x 72
            nn.MaxPool2d(2, 2),

            # N x 144 x 72 x 72 --> N x 72 x 72 x 72
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

            # N x 72 x 72 x 72 --> N x 72 x 36 x 36
            nn.MaxPool2d(2, 2),

            # N x 72 x 36 x 36 --> N x 36 x 36 x 36
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

            # N x 36 x 36 x 36 --> N x 36 x 18 x 18
            nn.MaxPool2d(2, 2),

            # N x 36 x 18 x 18 --> N x 36 x 18 x 18
            nn.Conv2d(
                in_channels=36,
                out_channels=36,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(36),
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            # N x 36 x 36 x 36 --> N x 72 x 36 x 36
            nn.Conv2d(
                in_channels=36,
                out_channels=72,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(72),
        )

        self.conv2 = nn.Sequential(
            # N x 72 x 72 x 72 --> N x 144 x 72 x 72
            nn.Conv2d(
                in_channels=72,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(144),
        )

        self.conv3 = nn.Sequential(
            # N x 144 x 145 x 145 --> N x 288 x 145 x 145
            nn.Conv2d(
                in_channels=144,
                out_channels=288,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(288),
        )

    def forward(self, x):
        # N x 36 x 18 x 18 --> N x 36 x 36 x 36
        out = interpolate(x, size=(36, 36))

        # N x 36 x 36 x 36 --> N x 72 x 36 x 36
        out = self.conv1(out)

        # N x 72 x 36 x 36 --> N x 72 x 72 x 72
        out = interpolate(out, size=(72, 72))

        # N x 72 x 72 x 72 --> N x 144 x 72 x 72
        out = self.conv2(out)

        # N x 144 x 72 x 72 --> N x 144 x 145 x 145
        out = interpolate(out, size=(145, 145))

        # N x 144 x 145 x 145 --> N x 288 x 145 x 145
        out = self.conv3(out)

        return out
