import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = nn.Sequential(
            # N x IN x H x W --> N x OUT x H x W
            nn.Conv2d(
                in_channels=288,
                out_channels=22,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.PReLU(22)
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(
            # N x IN x H x W --> N x OUT x H x W
            nn.Conv2d(
                in_channels=22,
                out_channels=288,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
        )

    def forward(self, h):
        y = self.decode(h)
        return y
