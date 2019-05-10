import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import itertools


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


def show_slices(slices,
                suptitle="Visualization",
                titles=('Saggital', 'Coronal', 'Axial'),
                figsize=(10, 5),
                fontsize=24):
    """ Function to display row of image slices """

    plt.rcParams["figure.figsize"] = figsize
    fig, axes = plt.subplots(1, len(slices))
    fig.suptitle(suptitle, fontsize=fontsize)
    for i, slc in enumerate(slices):
        axes[i].set_title(titles[i])
        axes[i].imshow(slc.T, cmap="gray", origin="lower")
    plt.tight_layout()


def count_model_parameters(model):
    """Counts total parameters of model."""

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total: {}, Trainable: {}".format(total, trainable))


def evaluate_adhd_classifier(classifier, rnn_encoder, criterion, dataloader, device):
    """Evaluate ADHD Classifier."""

    all_labels, all_preds = [], []

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

            all_labels.append(y.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

            running_loss += loss.item() * y.size(0)
            running_corrects += torch.sum(preds == y)

            total += y.size(0)

        avg_loss = running_loss / total
        acc = running_corrects.double() / total

        print("Total examples: {}, Loss: {:.5f}, Accuracy: {:.5f}".format(total, avg_loss, acc))

        all_labels = np.hstack(all_labels)
        all_preds = np.hstack(all_preds)

        return all_labels, all_preds


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


def get_np_dataset(rnn_encoder, dataloader, device):
    rnn_encoder.eval()
    features_array = []
    y_array = []

    count = 1
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            features = rnn_encoder(x).detach().cpu().view(-1).numpy()
            features_array.append(list(features))
            y_array.append(y.item())

            print(count)
            count += 1

        return np.array(features_array), np.array(y_array)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Credits: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def pooled_mean(mu_x, n_x, mu_y, n_y):
    """Calculates pooled mean.
    Args:
        mu_x: mean of group x.
        n_x: number of elements in group x.
        mu_y: mean of group y.
        n_y: number of elements in group y.
    Returns:
        Pooled mean and total number of elements.
    """
    n = n_y + n_x
    s_x = n_x / n
    s_y = n_y / n
    mu = s_x * mu_x + s_y * mu_y

    return mu, n

