import torch.nn as nn


class ConvDecoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.decode = nn.Sequential(
            #  N x 128 x 11 x 15 x 13 --> N x 64 x 22 x 29 x 25
            nn.ConvTranspose3d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(1, 0, 0)
            ),
            nn.ReLU(),
            nn.BatchNorm3d(128),

            # N x 64 x 22 x 29 x 25 --> N x 32 x 44 x 58 x 50
            nn.ConvTranspose3d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(1, 1, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm3d(128),


            # N x 32 x 44 x 58 x 50 --> N x 16 x 88 x 115 x 100
            nn.ConvTranspose3d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(1, 0, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm3d(128),

            # N x 16 x 88 x 115 x 100 --> N x out_channels x 176 x 229 x 200
            nn.ConvTranspose3d(
                in_channels=128,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(1, 0, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),

            # to remove checkerboard artifacts
            # layer to remove checkerboard artifacts
            #  N x out_channels x 176 x 229 x 200 -->   N x out_channels x 176 x 229 x 200
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

    def forward(self, x):
        out = self.decode(x)
        return out
