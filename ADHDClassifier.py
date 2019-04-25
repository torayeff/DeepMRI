import torch.nn as nn


class ADHDClassifier(nn.Module):
    def __init__(self, feature_channels, w, h, d):
        super().__init__()

        self.hidden = feature_channels * w * h * d
        self.fc1 = nn.Linear(self.hidden, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = x.view(-1, self.hidden)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x
