import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, h):
        super().__init__()

        self.encode = nn.Sequential(
            nn.Linear(288, h, bias=True),
            # nn.Sigmoid()
            nn.Tanh()
            # nn.ReLU()
            # nn.PReLU(h)
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class Decoder(nn.Module):
    def __init__(self, h):
        super().__init__()

        self.decode = nn.Sequential(
            nn.Linear(h, 288, bias=True)
        )

    def forward(self, h):
        y = self.decode(h)
        return y
