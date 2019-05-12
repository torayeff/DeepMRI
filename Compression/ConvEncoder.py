import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.encode = nn.Sequential(
            # N x input_channels x 176 x 229 x 200 --> N x 16 x 88 x 115 x 100
            nn.Conv3d(
                in_channels=input_channels,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm3d(16),

            # N x 16 x 88 x 115 x 100 --> N x 32 x 44 x 58 x 50
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm3d(32),

            # N x 32 x 44 x 58 x 50 --> N x 64 x 22 x 29 x 25
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm3d(64),

            # N x 64 x 22 x 29 x 25 --> N x 128 x 11 x 15 x 13
            nn.Conv3d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm3d(128),
        )

    def forward(self, x):
        out = self.encode(x)
        return out
