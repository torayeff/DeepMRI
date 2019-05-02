import torch
import torch.nn as nn


class RNNDecoder(nn.Module):
    def __init__(self,
                 rnn,
                 input_channels,
                 hidden_channels,
                 kernel_size,
                 stride,
                 padding,
                 hidden_kernel_size,
                 output_channels,
                 output_kernel_size,
                 bias=True):
        super().__init__()

        self.output_channels = output_channels

        self.rnn = rnn(input_channels + hidden_channels,
                       hidden_channels,
                       kernel_size,
                       stride,
                       padding,
                       hidden_kernel_size,
                       bias=bias
                       )

        # output volume
        output_padding = (output_kernel_size - 1) // 2
        self.out_conv = nn.Conv3d(
            in_channels=input_channels + 2*hidden_channels,
            out_channels=output_channels,
            kernel_size=output_kernel_size,
            stride=1,
            padding=output_padding,
            bias=bias
        )

    def forward(self, input_batch, hidden_batch, context_batch):
        """
        Decodes input_batch and context into prediction tensor.
        For this decoder implementation seq_len = 1.

        Args:
            input_batch: 5D tensor of shape (batch_size, input_channel, W, H, D)
            hidden_batch: 5D tensor of shape (batch_size, hidden_channel, W, H, D)
            context_batch: 5D tensor of shape (batch_size, context_channel, W, H, D)

        Returns:
            hidden tensor of 5D shape (batch_size, hidden_channel, W, H, D)
            prediction tensor of 5D shape (batch_size, input_channel, W, H, D)
        """

        input_concat = torch.cat((input_batch, context_batch), dim=1)

        hidden_batch = self.rnn(input_concat, hidden_batch)

        pred_concat = torch.cat((input_batch, context_batch, hidden_batch), dim=1)
        prediction_batch = self.out_conv(pred_concat)

        return prediction_batch, hidden_batch

