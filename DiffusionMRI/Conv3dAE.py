import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = nn.Sequential(
            # N x 7 x H x W x D --> N x 7 x H x W x D
            nn.Conv3d(
                in_channels=7,
                out_channels=7,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.ReLU(),

            nn.Conv3d(
                in_channels=7,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(
            # N x 7 x H x W x D --> N x 7 x H x W x D
            nn.Conv3d(
                in_channels=3,
                out_channels=7,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            )
        )

    def forward(self, h):
        y = self.decode(h)
        return y
