import torch.nn as nn
import math


class Conv3DGRUCell(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size,
                 stride,
                 padding,
                 hidden_kernel_size,
                 bias=True):

        super().__init__()

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # input convolutions
        self.conv_input = nn.Conv3d(
            in_channels=input_channels,
            out_channels=3*hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        # hidden convolutions
        # always stride=1 (keep spatial dimensions)
        hidden_padding = (hidden_kernel_size - 1) // 2
        self.conv_hidden = nn.Conv3d(
            in_channels=hidden_channels,
            out_channels=3*hidden_channels,
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=hidden_padding,
            bias=bias
        )

    def forward(self, input_batch, hidden=None):
        if hidden is None:
            # get spatial dimensions after input to hidden convolution
            dim1 = math.floor((input_batch.shape[2] - self.kernel_size + 2 * self.padding) / self.stride + 1)
            dim2 = math.floor((input_batch.shape[3] - self.kernel_size + 2 * self.padding) / self.stride + 1)
            dim3 = math.floor((input_batch.shape[4] - self.kernel_size + 2 * self.padding) / self.stride + 1)

            hidden = input_batch.new_zeros(input_batch.shape[0], self.hidden_channels, dim1, dim2, dim3,
                                           requires_grad=False)

        h_prev = hidden

        i_gates = self.conv_input(input_batch)
        h_gates = self.conv_hidden(hidden)

        ir_conv, iz_conv, in_conv = i_gates.chunk(3, dim=1)
        hr_conv, hz_conv, hn_conv = h_gates.chunk(3, dim=1)

        r = nn.Sigmoid()(ir_conv + hr_conv)
        z = nn.Sigmoid()(iz_conv + hz_conv)
        n = nn.Tanh()(in_conv + r * hn_conv)
        h = (1 - z) * n + z * h_prev

        return h
