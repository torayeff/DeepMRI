import torch.nn as nn


class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            # N x 1 x D x H x W --> N x 16 x D/2 x H/2 x W/2
            nn.Conv3d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # N x 16 x D/2 x H/2 x W/2 --> N x 32 x D/4 x H/4 x W/4
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # N x 32 x D/4 x H/4 x W/4 --> N x 32 x D/8 x H/8 x W/8
            nn.Conv3d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.d_conv1 = nn.Sequential(
            # N x 32 x D/8 x H/8 x W/8 --> N x 32 x D/4 x H/4 x W/4
            nn.Conv3d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )

        self.d_conv2 = nn.Sequential(
            # N x 32 x D/4 x H/4 x W/4 --> N x 16 x D/2 x H/2 x W/2
            nn.Conv3d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
        )

        self.d_conv3 = nn.Sequential(
            # N x 16 x D/2 x H/2 x W/2 --> N x 1 x D x H x W
            nn.Conv3d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.d_conv1(out)
        out = nn.functional.interpolate(out, scale_factor=2, mode='trilinear', align_corners=False)
        out = self.d_conv2(out)
        out = nn.functional.interpolate(out, scale_factor=2, mode='trilinear', align_corners=False)
        out = self.d_conv3(out)
        out = nn.functional.interpolate(out, scale_factor=2, mode='trilinear', align_corners=False)
        return out
