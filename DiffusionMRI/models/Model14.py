import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, num_features, num_hidden_1, num_latent, device):
        super().__init__()

        self.hidden_1 = nn.Linear(num_features, num_hidden_1)
        self.z_mean = nn.Linear(num_hidden_1, num_latent)
        self.z_log_var = nn.Linear(num_hidden_1, num_latent)
        self.device = device

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(self.device)
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        return z

    def forward(self, features):
        x = self.hidden_1(features)
        x = nn.LeakyReLU(negative_slope=0.0001)(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return z_mean, z_log_var, encoded


class Decoder(nn.Module):
    def __init__(self, num_features, num_hidden_1, num_latent):
        super().__init__()

        self.linear_3 = nn.Linear(num_latent, num_hidden_1)
        self.linear_4 = nn.Linear(num_hidden_1, num_features)

    def forward(self, encoded):
        x = self.linear_3(encoded)
        x = nn.LeakyReLU(negative_slope=0.0001)(x)
        x = self.linear_4(x)
        decoded = nn.Sigmoid()(x)
        return decoded
