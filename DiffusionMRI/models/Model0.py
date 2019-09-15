import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            # block 1
            nn.Conv2d(
                in_channels=288,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=144,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=144,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            # block 2
            nn.Conv2d(
                in_channels=144,
                out_channels=72,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=72,
                out_channels=72,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=72,
                out_channels=72,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            # block 3
            nn.Conv2d(
                in_channels=72,
                out_channels=36,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=36,
                out_channels=36,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=36,
                out_channels=36,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.block4 = nn.Sequential(
            # block 4
            nn.Conv2d(
                in_channels=36,
                out_channels=18,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=18,
                out_channels=18,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=18,
                out_channels=18,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block5 = nn.Sequential(
            # block 5
            nn.Conv2d(
                in_channels=18,
                out_channels=9,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=9,
                out_channels=9,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=9,
                out_channels=9,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            )
        )

    def forward(self, x, return_all=False):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            # block 1
            nn.ConvTranspose2d(
                in_channels=9,
                out_channels=9,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=1
            ),
            nn.Conv2d(
                in_channels=9,
                out_channels=18,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=18,
                out_channels=18,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=18,
                out_channels=18,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            )
        )

        self.block2 = nn.Sequential(
            # block 2
            nn.ConvTranspose2d(
                in_channels=18,
                out_channels=18,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=1
            ),
            nn.Conv2d(
                in_channels=18,
                out_channels=36,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=36,
                out_channels=36,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=36,
                out_channels=36,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True
            )
        )

        self.block3 = nn.Sequential(
            # block 3
            nn.ConvTranspose2d(
                in_channels=36,
                out_channels=36,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0
            ),
            nn.Conv2d(
                in_channels=36,
                out_channels=72,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=72,
                out_channels=72,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=72,
                out_channels=72,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True
            )
        )

        self.block4 = nn.Sequential(
            # block 4
            nn.ConvTranspose2d(
                in_channels=72,
                out_channels=72,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0
            ),
            nn.Conv2d(
                in_channels=72,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=144,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=144,
                out_channels=144,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True
            )
        )

        self.block5 = nn.Sequential(
            # block 5
            nn.ConvTranspose2d(
                in_channels=144,
                out_channels=144,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=1
            ),
            nn.Conv2d(
                in_channels=144,
                out_channels=288,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=288,
                out_channels=288,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=288,
                out_channels=288,
                kernel_size=3,
                stride=1,
                padding=2,
                bias=True
            )
        )

    def forward(self, h):
        out = self.block1(h)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        return out
