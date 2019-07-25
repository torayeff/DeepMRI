import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(
                in_channels=288,
                out_channels=22,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.PReLU(22)
        )

        self.input_size = input_size

    def forward(self, x, return_all=False):
        out = self.encode(x)
        return out


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(
            nn.Conv2d(
                in_channels=22,
                out_channels=288,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
        )

    def forward(self, h):
        y = self.decode(h)
        return y
