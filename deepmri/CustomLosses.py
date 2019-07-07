import torch.nn as nn


class MaskedLoss(nn.Module):
    def __init__(self, criterion=nn.MSELoss(reduction='sum')):
        super(MaskedLoss, self).__init__()
        self.criterion = criterion

    def forward(self, input_batch, output_batch, mask):
        x = input_batch * mask
        y = output_batch * mask

        bs = x.shape[0]
        channels = x.shape[1]

        avg_loss = 0.0
        for b in range(bs):
            r = mask[b].sum() * channels  # number of points in regions
            sample_loss = self.criterion(y[b], x[b]) / r
            avg_loss = avg_loss + sample_loss

        avg_loss = avg_loss / bs
        return avg_loss
