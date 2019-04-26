import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import time


def calc_conv_dim(w, k, s, p):
    """Calculates output dimensions of convolution operator."""

    dim = ((w + 2*p - k) / s) + 1
    print("Conv dim: ", dim, math.floor(dim))


def calc_transpose_conv_dim(w, k, s, p, out_p):
    """Calculates output dimensions of transpose convolution operator."""

    dim = (w - 1) * s - 2*p + k + out_p
    print("Deconv dim: ", dim, math.floor(dim))


def pad_3d(img_to_pad, target_dims=(256, 256, 256)):
    """Pads given 3D image."""

    diffs = np.array(target_dims) - np.array(img_to_pad.shape)
    pads = tuple([(d // 2 + (d % 2), d // 2) for d in diffs])
    return np.pad(img_to_pad, pads, 'constant')


def show_slices(slices):
    """ Function to display row of image slices """
    
    fig, axes = plt.subplots(1, len(slices))
    for i, slc in enumerate(slices):
        axes[i].imshow(slc.T, cmap="gray", origin="lower")


def count_model_parameters(model):
    """Counts total parameters of model."""

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total: {}, Trainable: {}".format(total, trainable))


def evaluate_adhd_classifier(classifier, rnn_encoder, criterion, dataloader, device):
    """Evaluate ADHD Classifier."""
    with torch.no_grad():
        running_loss = 0.0
        running_corrects = 0
        rnn_encoder.eval()
        classifier.eval()

        total = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            features = rnn_encoder(x)

            outputs = classifier(features)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, y)

            running_loss += loss.item() * y.size(0)
            running_corrects += torch.sum(preds == y)

            total += y.size(0)

        avg_loss = running_loss / total
        acc = running_corrects.double() / total

        print("Total examples: {}, Loss: {:.5f}, Accuracy: {:.5f}".format(total, avg_loss, acc))


def evaluate_rnn_encoder_decoder(encoder, decoder, criterion, dataloader, device):
    """Evaluates RNN AE."""
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        running_loss = 0.0

        total = 0
        for data in dataloader:

            # -------------------Seq2Seq Start------------------ #

            src_batch = data.to(device)
            trg_batch = data.to(device)

            seq_len = trg_batch.shape[1]

            context_batch = encoder(src_batch)

            hidden_batch = context_batch

            # first input is <sos> in learning phrase representation
            # in this case it is tensor of ones
            input_batch = trg_batch.new_ones(trg_batch[:, 0, :, :, :, :].shape)

            outputs = trg_batch.new_zeros(trg_batch.shape)

            for t in range(seq_len):
                input_batch, hidden_batch = decoder(input_batch, hidden_batch, context_batch)
                outputs[:, t, :, :, :, :] = input_batch

            loss = criterion(trg_batch, outputs)
            # -------------------Seq2Seq End------------------- #

            running_loss += loss.item() * data.size(0)
            total += data.size(0)

        print("Average loss: {}".format(running_loss / total))
