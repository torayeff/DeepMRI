import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import itertools
import pandas as pd
import os
import nibabel as nib
import deepmri.Datasets as Datasets


def calc_conv_dim(w, k, s, p):
    """Calculates output dimensions of convolution operator."""

    dim = ((w + 2 * p - k) / s) + 1
    print("Conv dim: ", dim, math.floor(dim))


def calc_transpose_conv_dim(w, k, s, p, out_p):
    """Calculates output dimensions of transpose convolution operator."""

    dim = (w - 1) * s - 2 * p + k + out_p
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
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show()


def show_one_slice(slc,
                   title="One Slice",
                   figsize=(10, 5),
                   fontsize=12):

    plt.rcParams["figure.figsize"] = figsize
    plt.imshow(slc.T, cmap="gray", origin="lower")
    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def count_model_parameters(model):
    """Counts total parameters of model."""

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


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


def pooled_mean_std(dataset, total_n):
    """Calculates mean and standard deviation for the training set.
    Args:
        dataset: pytorch dataset.
        total_n: Total number of voxels.
    Returns:
        Mean and standard deviation, total voxels for verification.
    """
    start = time.time()

    print("Total examples in dataset: {}".format(len(dataset)))

    mu, n = 0, 0
    sqr_sum = 0.0
    temp_sqr_sum = 0.0  # temporary squared sum

    for c, data in enumerate(dataset, 1):
        print("Processing #{}: ".format(c), end="")

        # calculate pooled mean
        mu_y = data.mean()
        n_y = 1
        for sh in data.shape:
            n_y *= sh
        mu, n = pooled_mean(mu, n, mu_y, n_y)
        print("Pooled mu={}, n={}".format(mu, n))

        # keep track of running squared sums
        temp_sqr_sum += (data ** 2).sum()
        if c % 10 == 0:
            sqr_sum += temp_sqr_sum / total_n
            temp_sqr_sum = 0.0  # reset

    sqr_sum += temp_sqr_sum / total_n
    std = np.sqrt(sqr_sum - (mu ** 2))

    end = time.time()
    print("Dataset mu={}, std={}, n={}".format(mu, std, n))
    print("Total calculation time: {:.5f}.".format(end - start))
    return mu, std, n


def create_orientation_dataset(csv_file,
                               save_dir,
                               orients=(True, True, True)):
    """Creates axial, coronal, sagittal volumes.

    Args:
        csv_file: csv file with dMRI paths.
        save_dir: Directory to save the data.
        orients: Orients to create.
    """
    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():
        if np.sum(orients) == 0:
            print("None of the orientation has been selected.")
            break

        print("Loading file: {}".format(row['dmri_path']))
        data = nib.load(row['dmri_path']).get_fdata()

        # sagittal orientation
        if orients[0]:
            print("Processing sagittal orientation...")
            for idx in range(data.shape[0]):
                sagittal = data[idx, :, :, :]
                check_sum = np.sum(sagittal)
                if check_sum == 0:
                    print("Sagittal idx={} is empty. Skipping.".format(idx))
                else:
                    save_path = "sagittal/data_{}_sagittal_idx_{}".format(row['subj_id'], idx)
                    save_path = os.path.join(save_dir, save_path)
                    np.savez(save_path, data=sagittal)

        # coronal orientation
        if orients[1]:
            print("Processing coronal orientation...")
            for idx in range(data.shape[1]):
                coronal = data[:, idx, :, :]
                check_sum = np.sum(coronal)
                if check_sum == 0:
                    print("Coronal idx={} is empty. Skipping.".format(idx))
                else:
                    save_path = "coronal/data_{}_coronal_idx_{}".format(row['subj_id'], idx)
                    save_path = os.path.join(save_dir, save_path)
                    np.savez(save_path, data=coronal)

        # axial orientation
        if orients[2]:
            print("Processing axial orientation...")
            for idx in range(data.shape[2]):
                axial = data[:, :, idx, :]
                check_sum = np.sum(axial)
                if check_sum == 0:
                    print("Axial idx={} is empty. Skipping.".format(idx))
                else:
                    save_path = "axial/data_{}_axial_idx_{}".format(row['subj_id'], idx)
                    save_path = os.path.join(save_dir, save_path)
                    np.savez(save_path, data=axial)
    print("Done!")


def batch_loss(dataloader, encoder, decoder, criterion, device):
    """Calculates average loss on whole dataset."""
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        total = 0
        running_loss = 0
        for data in dataloader:

            x = data.to(device)

            out = decoder(encoder(x))

            loss = criterion(x, out)

            running_loss = running_loss + loss.item() * data.size(0)
            total += data.size(0)

        avg_loss = running_loss / total

    return avg_loss, total


def evaluate_ae(x, encoder, decoder, criterion, device, mu, std, t, plot=True, add_channel=False):
    """Evaluates AE."""
    encoder.eval()
    decoder.eval()

    # x is numpy array with dim: C x W x H
    if add_channel:
        x = x.transpose(1, 2, 0)[np.newaxis, ...]  # 1 x W x H x C

    x = (x - mu) / std
    x = torch.tensor(x).float().unsqueeze(0)

    with torch.no_grad():
        x = x.to(device)
        y = decoder(encoder(x))

    x = x * std + mu
    # unnormalize
    y = y * std + mu
    y = y.clamp(min=0)

    loss = criterion(x, y)

    x = x.squeeze().cpu().numpy()
    y = y.squeeze().cpu().numpy()

    if add_channel:
        x = x.transpose(2, 0, 1)
        y = y.transpose(2, 0, 1)

    if plot:
        suptitle = "Loss: {}".format(loss)
        original_title = "Original\n Minval: {}, Maxval: {}".format(x.min(), x.max())
        recons_title = "Reconstruction\n Minval: {}, Maxval: {}".format(y.min(), y.max())
        show_slices([
            x[t, :, :],
            y[t, :, :]
        ], suptitle=suptitle, titles=[original_title, recons_title], fontsize=16)

    return y, loss.item()


def train_ae(encoder,
             decoder,
             criterion,
             optimizer,
             device,
             trainloader,
             num_epochs,
             model_name,
             experiment_dir,
             checkpoint=1
             ):
    """Trains AutoEncoder."""

    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()
        epoch_start = time.time()
        total_examples = 0
        running_loss = 0.0

        for data in trainloader:
            iter_time = time.time()

            # zero gradients
            optimizer.zero_grad()

            # forward
            x = data.to(device)
            out = decoder(encoder(x))

            # calculate loss
            loss = criterion(x, out)

            # backward
            loss.backward()

            # update params
            optimizer.step()

            # track loss
            running_loss = running_loss + loss.item() * data.size(0)
            total_examples += data.size(0)
            # print("Iter time: {}".format(time.time() - iter_time))

        if epoch % checkpoint == 0:
            torch.save(encoder.state_dict(), "{}models/{}_encoder_epoch_{}".format(experiment_dir, model_name, epoch))
            torch.save(decoder.state_dict(), "{}models/{}_decoder_epoch_{}".format(experiment_dir, model_name, epoch))

        epoch_loss = running_loss / total_examples
        print("Epoch #{}/{},  epoch loss: {}, epoch time: {:.5f} seconds".format(epoch,
                                                                                 num_epochs,
                                                                                 epoch_loss,
                                                                                 time.time() - epoch_start))

