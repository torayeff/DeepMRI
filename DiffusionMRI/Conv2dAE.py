import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.encode = nn.Sequential(
            # N x IN x H x W --> N x OUT x H x W
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class ConvDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.decode = nn.Sequential(
            # N x IN x H x W --> N x OUT x H x W
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
        )

    def forward(self, h):
        y = self.decode(h)
        return y
