import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(1),
            nn.PReLU()
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class ConvDecoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(90),
            nn.PReLU()
        )

    def forward(self, x):
        out = self.decode(x)
        return out
