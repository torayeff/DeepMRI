import torch.nn as nn
from Conv3DLSTMCell import Conv3DLSTMCell


class Encoder(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size,
                 stride,
                 padding,
                 hidden_kernel_size,
                 bias=True):
        super().__init__()

        self.lstm_cell = Conv3DLSTMCell(input_channels,
                                        hidden_channels,
                                        kernel_size,
                                        stride,
                                        padding,
                                        hidden_kernel_size,
                                        bias=bias
                                        )

    def forward(self, input_batch):
        """
        Encodes sequence to 2D representation.
        Args:
            input_batch: 6D tensor (batch, time, channel, x, y, z)
        Returns:
            Last hidden and cell state
        """

        h, c = (None, None)
        seq_len = input_batch.shape[1]
        for t in range(seq_len):
            h, c = self.lstm_cell(input_batch[:, t, :, :, :, :], (h, c))

        return h, c
