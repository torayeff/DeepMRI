import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.encode = nn.Sequential(
            # N x 288 x 145 x 174 --> N x 288 x 73 x 87
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=288,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(16, 288),
            nn.ReLU(),

            # N x 288 x 73 x 87 --> N x 144 x 37 x 44
            nn.Conv2d(
                in_channels=288,
                out_channels=144,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(12, 144),
            nn.ReLU(),

            # N x 144 x 37 x 44 --> N x 128 x 19 x 22
            nn.Conv2d(
                in_channels=144,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, 128),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class ConvDecoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.decode = nn.Sequential(
            #  N x 128 x 19 x 22--> N x 144 x 37 x 44
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=144,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 1),
                bias=False
            ),
            nn.GroupNorm(12, 144),
            nn.ReLU(),

            # N x 144 x 37 x 44 --> N x 288 x 73 x 87
            nn.ConvTranspose2d(
                in_channels=144,
                out_channels=288,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 0),
                bias=False
            ),
            nn.GroupNorm(16, 288),
            nn.ReLU(),

            # N x 288 x 73 x 87 --> N x 288 x 145 x 174
            nn.ConvTranspose2d(
                in_channels=288,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 1),
                bias=True
            )
        )

    def forward(self, x):
        out = self.decode(x)
        return out
