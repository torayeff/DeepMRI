import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.encode = nn.Sequential(
            # N x 1 x 145 x 174 x 288 --> N x 2 x 73 x 87 x 144
            nn.Conv3d(
                in_channels=input_channels,
                out_channels=2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm3d(2),

            # N x 2 x 73 x 87 x 144 --> N x 4 x 37 x 44 x 72
            nn.Conv3d(
                in_channels=2,
                out_channels=4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm3d(4),

            # N x 4 x 37 x 44 x 72 --> N x 4 x 19 x 22 x 36
            nn.Conv3d(
                in_channels=4,
                out_channels=4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm3d(4),
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class ConvDecoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.decode = nn.Sequential(
            #  N x 4 x 19 x 22 x 36 --> N x 4 x 37 x 44 x 72
            nn.ConvTranspose3d(
                in_channels=4,
                out_channels=4,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 1, 1),
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm3d(4),

            # N x 4 x 37 x 44 x 72 --> N x 2 x 73 x 87 x 144
            nn.ConvTranspose3d(
                in_channels=4,
                out_channels=2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 0, 1),
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm3d(2),

            # N x 2 x 73 x 87 x 144 --> N x 1 x 145 x 174 x 288
            nn.ConvTranspose3d(
                in_channels=2,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 1, 1),
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        out = self.decode(x)
        return out
