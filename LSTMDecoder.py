import torch.nn as nn
from Conv3DLSTMCell import Conv3DLSTMCell


class LSTMDecoder(nn.Module):
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

    def forward(self, input_batch, hidden, seq_len=300):
        """
        Decodes latent representation to the sequence of 3D images.
        Args:
            input_batch: 5D tensor (batch, channel, x, y, z)
            hidden: hidden and cell states from encoder
            seq_len: sequence length
        Returns:
            outputs
        """

        h, c = hidden
        for t in range(seq_len):
            h, c = self.lstm_cell(input_batch[:, t, :, :, :, :], (h, c))

        return h, c
