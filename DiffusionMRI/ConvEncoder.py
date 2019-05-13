import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.encode = nn.Sequential(
            # N x 1 x 145 x 174 x 145 --> N x 16 x 73 x 87 x 73
            nn.Conv3d(
                in_channels=input_channels,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm3d(16),

            # N x 16 x 73 x 87 x 73 --> N x 32 x 37 x 44 x 37
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm3d(32),

            # N x 32 x 37 x 44 x 37 --> N x 64 x 19 x 22 x 19
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm3d(64),
        )

    def forward(self, x):
        out = self.encode(x)
        return out
