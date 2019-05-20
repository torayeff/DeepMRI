import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.encode = nn.Sequential(
            # N x 288 x 174 x 145 --> N x 288 x 87 x 73
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=288,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(288),

            # N x 288 x 87 x 73 --> N x 144 x 44 x 37
            nn.Conv2d(
                in_channels=288,
                out_channels=144,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(144),

            # N x 144 x 44 x 37 --> N x 128 x 22 x 19
            nn.Conv2d(
                in_channels=144,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class ConvDecoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.decode = nn.Sequential(
            #  N x 128 x 22 x 19 --> N x 144 x 44 x 37
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=144,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(1, 0),
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(144),

            # N x 144 x 44 x 37 --> N x 288 x 87 x 73
            nn.ConvTranspose2d(
                in_channels=144,
                out_channels=288,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 0),
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(288),

            # N x 288 x 87 x 73 --> N x 288 x 174 x 145
            nn.ConvTranspose2d(
                in_channels=288,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(1, 0),
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.decode(x)
        return out
