import torch.nn as nn


class Conv3DLSTMCell(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size,
                 stride,
                 padding,
                 hidden_kernel_size,
                 bias=True):

        super().__init__()

        # input to hidden convolution for i, f, o, g all together
        self.conv_ih = nn.Conv3d(
            in_channels=input_channels,
            out_channels=4*hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        ),

        # hidden to hidden convolution for i, f, o, g all together
        # always stride=1 (keep spatial dimensions)
        hidden_padding = (hidden_kernel_size - 1) // 2
        self.conv_hh = nn.Conv3d(
            in_channels=hidden_channels,
            out_channels=4*hidden_channels,
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=hidden_padding,
            dilation=1,
            bias=bias
        )

    def forward(self, input, hidden):
        h_prev, c_prev = hidden
        gates = self.conv_ih(input) + self.conv_hh(h_prev)

        # dim=0 is batch, dim=1 is channel
        i, f, o, g = gates.chunk(4, dim=1)

        i = nn.Sigmoid()(i)
        f = nn.Sigmoid()(f)
        o = nn.Sigmoid()(o)
        g = nn.Tanh()(g)

        c_cur = (f * c_prev) + (i * g)
        h_cur = o * nn.Tanh()(c_cur)

        return h_cur, c_cur
