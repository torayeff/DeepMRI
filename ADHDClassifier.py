import torch.nn as nn
from RNNEncoder import RNNEncoder
from Conv3DRNNCell import Conv3DGRUCell


class ADHDClassifier(nn.Module):
    def __init__(self, rnn_encoder, feature_channels, w, h, d):
        super().__init__()

        self.rnn_encoder = rnn_encoder
        self.hidden = feature_channels * w * h * d
        self.fc1 = nn.Linear(self.hidden, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.rnn_encoder(x)
        x = x.view(-1, self.hidden)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        return x
