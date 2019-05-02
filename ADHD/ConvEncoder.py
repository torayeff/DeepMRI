import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.encode = nn.Sequential(
            # N x 1 x 49 x 58 x 47 --> N x 16 x 25 x 29 x 24
            nn.Conv3d(
                in_channels=input_channels,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            # N x 16 x 25 x 29 x 24 --> N x 32 x 13 x 15 x 12
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            # N x 32 x 13 x 15 x 12 --> N x 64 x 7 x 8 x 6
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.encode(x)
        return out
