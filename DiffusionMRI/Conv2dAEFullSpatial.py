import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.encode = nn.Sequential(
            # N x 288 x 174 x 145 --> N x 144 x 174 x 145
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(12, 144),
            nn.ReLU(),

            # N x 144 x 174 x 145 --> N x 72 x 174 x 145
            nn.Conv2d(
                in_channels=144,
                out_channels=72,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, 72),
            nn.ReLU(),

            # N x 72 x 174 x 145 --> N x 36 x 174 x 145
            nn.Conv2d(
                in_channels=72,
                out_channels=36,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(6, 36),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class ConvDecoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.decode = nn.Sequential(
            #  N x 36 x 174 x 145 --> N x 72 x 174 x 145
            nn.Conv2d(
                in_channels=36,
                out_channels=72,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(8, 72),
            nn.ReLU(),

            # N x 72 x 174 x 145 --> N x 144 x 174 x 145
            nn.Conv2d(
                in_channels=72,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(12, 144),
            nn.ReLU(),

            # N x 144 x 174 x 145 --> N x 288 x 174 x 145
            nn.Conv2d(
                in_channels=144,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            )
        )

    def forward(self, x):
        out = self.decode(x)
        return out
