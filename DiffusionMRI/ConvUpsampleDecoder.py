import torch.nn as nn
from torch.nn.functional import interpolate


class ConvUpsampleDecoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            #  N x 64 x 37 x 44 x 37 --> N x 32 x 37 x 44 x 37
            nn.ConvTranspose3d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            # N x 32 x 73 x 87 x 73 --> N x 16 x 73 x 87 x 73
            nn.ConvTranspose3d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            # N x 16 x 145 x 174 x 145 --> N x 1 x 145 x 174 x 145
            nn.ConvTranspose3d(
                in_channels=16,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm3d(1)
        )

    def forward(self, x):
        #  N x 64 x 19 x 22 x 19 --> N x 64 x 37 x 44 x 37
        x = interpolate(x, size=(37, 44, 37))

        #  N x 64 x 37 x 44 x 37 --> N x 32 x 37 x 44 x 37
        x = self.conv1(x)

        # N x 32 x 37 x 44 x 37 --> N x 32 x 73 x 87 x 73
        x = interpolate(x, size=(73, 87, 73))

        # N x 32 x 73 x 87 x 73 --> N x 16 x 73 x 87 x 73
        x = self.conv2(x)

        # N x 16 x 73 x 87 x 73 --> N x 16 x 145 x 174 x 145
        x = interpolate(x, size=(145, 174, 145))

        # N x 16 x 145 x 174 x 145 --> N x 1 x 145 x 174 x 145
        x = self.conv3(x)

        return x
