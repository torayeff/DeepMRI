import torch.nn as nn


class ConvTransposeDecoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.decode = nn.Sequential(
            #  N x 64 x 19 x 22 x 19 --> N x 32 x 37 x 44 x 37
            nn.ConvTranspose3d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 1, 0)
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            # N x 32 x 37 x 44 x 37 --> N x 16 x 73 x 87 x 73
            nn.ConvTranspose3d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 0, 0)
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            # N x 16 x 73 x 87 x 73 --> N x 1 x 145 x 174 x 145
            nn.ConvTranspose3d(
                in_channels=16,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 1, 0)
            ),
            nn.BatchNorm3d(1),
            nn.ReLU(),

            # layer to remove checkerboard artifacts
            # N x 1 x 145 x 174 x 145 -->  N x 1 x 145 x 174 x 145
            nn.Conv3d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

    def forward(self, x):
        out = self.decode(x)
        return out
