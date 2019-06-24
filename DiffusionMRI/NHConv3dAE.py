import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = nn.Sequential(
            # N x 288 x H x W x D --> N x 7 x 1 x 1 x 1
            nn.Conv3d(
                in_channels=288,
                out_channels=288,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.LeakyReLU()
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(
            # N x 7 x 1 x 1 x 1 --> N x 288 x H x W x D
            nn.ConvTranspose3d(
                in_channels=288,
                out_channels=288,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            )
        )

    def forward(self, h):
        y = self.decode(h)
        return y
