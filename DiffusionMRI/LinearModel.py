import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = nn.Sequential(
            nn.Linear(288, 88, bias=False),
            nn.PReLU(88),
            nn.Linear(88, 44, bias=True),
            nn.PReLU(44),
            nn.Linear(44, 22, bias=True),
            nn.PReLU(22)
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(
            nn.Linear(22, 44, bias=True),
            nn.PReLU(44),
            nn.Linear(44, 88, bias=True),
            nn.PReLU(88),
            nn.Linear(88, 288, bias=True)
        )

    def forward(self, h):
        y = self.decode(h)
        return y
