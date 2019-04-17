import torch.nn as nn
from Conv3DGRUCell import Conv3DGRUCell


class GRUEncoder(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size,
                 stride,
                 padding,
                 hidden_kernel_size,
                 bias=True):
        super().__init__()

        self.rnn = Conv3DGRUCell(input_channels,
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
            input_batch: 6D tensor (batch, time, channel, x, y, z)
        Returns:
            Last hidden state.
        """

        hidden = None
        seq_len = input_batch.shape[1]
        for t in range(seq_len):
            hidden = self.rnn(input_batch[:, t, :, :, :, :], hidden)

        return hidden
