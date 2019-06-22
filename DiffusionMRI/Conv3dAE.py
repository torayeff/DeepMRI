import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = nn.Sequential(
            # N x 288 x H x W x D --> N x 36 x H x W x D
            nn.Conv3d(
                in_channels=64,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
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
            # N x 36 x H x W x D --> N x 288 x H x W x D
            nn.Conv3d(
                in_channels=6,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
        )

    def forward(self, h):
        y = self.decode(h)
        return y
