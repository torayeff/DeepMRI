import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = nn.Sequential(
            nn.Linear(288, 100),
            nn.PReLU(100),
            nn.Linear(100, 50),
            nn.PReLU(50),
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(
            nn.Linear(50, 288)
        )

    def forward(self, h):
        y = self.decode(h)
        return y
