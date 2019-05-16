import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.encode = nn.Sequential(
            # N x 288 x 145 x 174 --> N x 16 x 73 x 87
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            # N x 16 x 73 x 87 --> N x 32 x 37 x 44
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # N x 32 x 37 x 44 --> N x 64 x 19 x 22
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class ConvDecoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.decode = nn.Sequential(
            #  N x 64 x 19 x 22--> N x 32 x 37 x 44
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 1),
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            # N x 32 x 37 x 44 --> N x 16 x 73 x 87
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 0),
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            # N x 16 x 73 x 87 --> N x 288 x 145 x 174
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 1),
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.decode(x)
        return out
