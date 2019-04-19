"""
Based on "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"
https://arxiv.org/abs/1406.1078
"""
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_batch, trg_batch):
        """
        Seq2Seq model.
        Args:
            src_batch: 6D tensor of shape (batch_size, time, channel, W, H, D)
            trg_batch: 6D tensor of shape (batch_size, time, channel, W, H, D)

        Returns:
            6D tensor of shape (batch_size, time, channel, W, H, D)
        """
        seq_len = trg_batch.shape[1]

        context_batch = self.encoder(src_batch)

        hidden_batch = context_batch

        # first input is <sos> in learning phrase represntation
        # in this case it is tensor of ones
        input_batch = trg_batch.new_ones(trg_batch[:, 0, :, :, :, :].shape)

        outputs = trg_batch.new_zeros(trg_batch.shape)

        for t in range(seq_len):
            input_batch, hidden_batch = self.decoder(input_batch, hidden_batch, context_batch)
            outputs[:, t, :, :, :, :] = input_batch

        return outputs
