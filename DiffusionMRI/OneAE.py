import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = nn.Sequential(
            nn.Linear(288, 288),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decode = nn.Sequential(
            nn.Linear(288, 288)
        )

    def forward(self, h):
        y = self.decode(h)
        return y
