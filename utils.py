import math
import numpy as np
import matplotlib.pyplot as plt


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
