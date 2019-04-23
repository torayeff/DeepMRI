import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,
                 rnn,
                 input_channels,
                 hidden_channels,
                 kernel_size,
                 stride,
                 padding,
                 hidden_kernel_size,
                 bias=True):
        super().__init__()

        self.rnn = rnn(input_channels,
                       hidden_channels,
                       kernel_size,
                       stride,
                       padding,
                       hidden_kernel_size,
                       bias=bias
                       )

    def forward(self, input_batch):
        """
        Encodes sequence to latent representation.
        Args:
            input_batch: 6D tensor (batch, time, input_channel, W, H, D)
        Returns:
            Last hidden state tensor of 5D shape (batch, hidden_channel, W, H, D).
        """

        hidden = None
        seq_len = input_batch.shape[1]
        for t in range(seq_len):
            hidden = self.rnn(input_batch[:, t, :, :, :, :], hidden)

        return hidden
