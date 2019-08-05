import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate


class ConvVAE(torch.nn.Module):

    def __init__(self, num_latent, device):
        super().__init__()

        self.device = device

        ###############
        # ENCODER
        ##############

        self.enc_conv_1 = nn.Conv2d(
            in_channels=288,
            out_channels=144,
            kernel_size=3,
            stride=2,
            padding=0
        )

        self.enc_conv_2 = nn.Conv2d(
            in_channels=144,
            out_channels=72,
            kernel_size=3,
            stride=2,
            padding=0
        )

        self.enc_conv_3 = nn.Conv2d(
            in_channels=72,
            out_channels=22,
            kernel_size=3,
            stride=2,
            padding=0
        )

        self.z_mean = torch.nn.Linear(22 * 17 * 17, num_latent)
        self.z_log_var = torch.nn.Linear(22 * 17 * 17, num_latent)

        ###############
        # DECODER
        ##############

        self.dec_linear_1 = torch.nn.Linear(num_latent, 22 * 17 * 17)

        self.dec_deconv_1 = nn.Conv2d(
            in_channels=22,
            out_channels=44,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.dec_deconv_2 = nn.Conv2d(
            in_channels=44,
            out_channels=88,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.dec_deconv_3 = nn.Conv2d(
            in_channels=88,
            out_channels=288,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(self.device)
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        return z

    def encoder(self, features):
        x = self.enc_conv_1(features)
        x = F.leaky_relu(x)

        x = self.enc_conv_2(x)
        x = F.leaky_relu(x)

        x = self.enc_conv_3(x)
        x = F.leaky_relu(x)

        z_mean = self.z_mean(x.view(-1, 22 * 17 * 17))
        z_log_var = self.z_log_var(x.view(-1, 22 * 17 * 17))
        encoded = self.reparameterize(z_mean, z_log_var)

        return z_mean, z_log_var, encoded

    def decoder(self, encoded, return_feature=False):
        x = self.dec_linear_1(encoded)
        x = x.view(-1, 22, 17, 17)

        x = interpolate(x, size=(145, 145), mode='bilinear', align_corners=True)

        # this part will be used for Random Forest classifier
        if return_feature:
            return x

        x = self.dec_deconv_1(x)
        x = F.leaky_relu(x)

        x = self.dec_deconv_2(x)
        x = F.leaky_relu(x)

        x = self.dec_deconv_3(x)
        x = F.leaky_relu(x)

        decoded = nn.Sigmoid()(x)
        return decoded

    def forward(self, features):
        z_mean, z_log_var, encoded = self.encoder(features)
        decoded = self.decoder(encoded)

        return z_mean, z_log_var, encoded, decoded
