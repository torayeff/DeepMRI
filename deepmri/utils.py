import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import itertools
import pandas as pd
import os
import nibabel as nib


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
                fontsize=24,
                cmap=None):
    """ Function to display row of image slices """

    plt.rcParams["figure.figsize"] = figsize
    fig, axes = plt.subplots(1, len(slices))
    fig.suptitle(suptitle, fontsize=fontsize)
    for i, slc in enumerate(slices):
        axes[i].set_title(titles[i])
        axes[i].imshow(slc.T, cmap=cmap, origin="lower")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()


def show_one_slice(slc,
                   title="One Slice",
                   figsize=(10, 5),
                   fontsize=12,
                   cmap=None):
    plt.rcParams["figure.figsize"] = figsize
    plt.imshow(slc.T, cmap=cmap, origin="lower")
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
        mu_y = data['data'].mean()
        n_y = np.prod(data['data'].shape)
        mu, n = pooled_mean(mu, n, mu_y, n_y)
        print("Pooled mu={}, n={}".format(mu, n))

        # keep track of running squared sums
        temp_sqr_sum += (data['data'] ** 2).sum()
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
                               orients=(0, 1, 2)):
    """Creates axial, coronal, sagittal volumes.

    Args:
        csv_file: csv file with dMRI paths.
        save_dir: Directory to save the data.
        orients: 0 - Sagittal, 1 - Coronal, 2 - Axial
    """
    orient_names = ['sagittal', 'coronal', 'axial']

    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():

        print("Loading file: {}".format(row['dmri_path']))
        data = nib.load(row['dmri_path']).get_fdata()  # 4D data: W x H x D x T

        for orient in orients:

            orient_name = orient_names[orient]
            print("Processing {} orientation...".format(orient_name))
            st = time.time()

            for idx in range(data.shape[orient]):
                volume = None
                if orient == 0:
                    volume = data[idx, :, :, :]
                elif orient == 1:
                    volume = data[:, idx, :, :]
                elif orient == 2:
                    volume = data[:, :, idx, :]

                check_sum = np.sum(volume)
                if check_sum == 0:
                    print("{} idx={} is empty. Skipping.".format(orient_name, idx))
                else:
                    save_path = "{}/data_{}_{}_idx_{}".format(orient_name, row['subj_id'], orient_name, idx)
                    save_path = os.path.join(save_dir, save_path)
                    volume = volume.transpose(2, 0, 1)  # channel x width x height)

                    np.savez(save_path, data=volume)

            print("Processed in {:.5f} seconds.".format(time.time() - st))
    print("Done!")


def dataset_performance(dataset,
                        encoder,
                        decoder,
                        criterion,
                        device,
                        mu,
                        std,
                        every_iter=10 ** 10,
                        eval_mode=True,
                        plot=False):
    """Calculates average loss on whole dataset."""

    if eval_mode:
        encoder.eval()
        decoder.eval()
    else:
        encoder.train()
        decoder.train()

    print("Encoder training mode: {}".format(encoder.training))
    print("Decoder training mode: {}".format(encoder.training))

    start = time.time()

    total_examples = len(dataset)

    min_loss = 10 ** 9
    max_loss = 0

    best_img = None
    worst_img = None

    print("Total examples: {}".format(total_examples))

    with torch.no_grad():
        running_loss = 0
        eval_start = time.time()
        for i, data in enumerate(dataset, 1):
            x = data['data'].unsqueeze(0).to(device)

            out = decoder(encoder(x))

            loss = criterion(x, out)

            if loss.item() < min_loss:
                min_loss = loss.item()
                best_img = data

            if loss.item() > max_loss:
                max_loss = loss.item()
                worst_img = data

            running_loss = running_loss + loss.item()

            if i % every_iter == 0:
                print("Evaluated {}/{} examples. Evaluation time: {:.5f} secs.".format(i,
                                                                                       total_examples,
                                                                                       time.time() - eval_start))
                eval_start = time.time()

        avg_loss = running_loss / total_examples

    print("Evaluated {}/{}, Total evaluation time: {:.5f} secs.".format(total_examples,
                                                                        total_examples,
                                                                        time.time() - start))
    print("Min loss: {:.5f}\nMax loss: {:.5f}\nAvg loss: {:.5f}\nBest Reconstruction: {}\nWorst Reconstruction: {}"
          .format(min_loss,
                  max_loss,
                  avg_loss,
                  best_img['file_name'],
                  worst_img['file_name']))

    if plot:
        t = np.random.randint(low=0, high=288)
        print("Showing slice at t={}".format(t))
        # show best reconstruction
        visualize_ae_results(best_img['data'],
                             encoder,
                             decoder,
                             criterion,
                             device,
                             mu,
                             std,
                             t,
                             suptitle='Best reconstruction.',
                             scale_back=True,
                             cmap='gray')
        # show worst reconstruction
        visualize_ae_results(worst_img['data'],
                             encoder,
                             decoder,
                             criterion,
                             device,
                             mu,
                             std,
                             t,
                             suptitle='Worst reconstruction',
                             scale_back=True,
                             cmap='gray')


def visualize_ae_results(x, encoder, decoder, criterion, device, mu, std, t,
                         scale_back=True,
                         suptitle="Visualization",
                         cmap=None):
    """Visualizes AE results."""
    encoder.eval()
    decoder.eval()

    # x is tensor with dim C x W x H
    x = x.unsqueeze(0)  # add batch dim

    with torch.no_grad():
        x = x.to(device)
        y = decoder(encoder(x))
        loss_before_scaling = criterion(x, y)

    if scale_back:
        x = x * std + mu
        y = y * std + mu
        y = y.clamp(min=x.min())
        y = y.clamp(max=x.max())
        loss_after_scaling = criterion(x, y)

    x = x.squeeze().cpu().numpy()
    y = y.squeeze().cpu().numpy()

    original_title = "Original\n Minval: {:.5f}, Maxval: {:.5f}".format(x.min(), x.max())
    recons_title = "Reconstruction\n Minval: {:.5f}, Maxval: {:.5f}".format(y.min(), y.max())
    show_slices([
        x[t, :, :],
        y[t, :, :]
    ], suptitle=suptitle, titles=[original_title, recons_title], fontsize=16, cmap=cmap)
    print("Loss: {}".format(loss_before_scaling))
    print("Loss after scaling back: {}".format(loss_after_scaling))
    print("-" * 100)


def evaluate_ae(encoder,
                decoder,
                criterion,
                device,
                trainloader,
                print_iter=False,
                ):
    """Evaluates AE."""

    start = time.time()
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        total_examples = 0
        running_loss = 0.0

        for i, batch in enumerate(trainloader, 1):
            x = batch['data'].to(device)
            out = decoder(encoder(x))

            # calculate loss
            loss = criterion(x, out)

            # track loss
            running_loss = running_loss + loss.item() * batch['data'].size(0)
            total_examples += batch['data'].size(0)

            if print_iter:
                print("Batch #{}, Batch loss: {}".format(i, loss.item()))

        avg_loss = running_loss / total_examples

    print("Evaluated {} examples, Avg. loss: {}, Time: {:.5f}".format(total_examples, avg_loss, time.time() - start))
    return avg_loss


def train_ae(encoder,
             decoder,
             criterion,
             optimizer,
             device,
             trainloader,
             num_epochs,
             model_name,
             experiment_dir,
             start_epoch=0,
             scheduler=None,
             checkpoint=1,
             print_iter=False,
             eval_epoch=5
             ):
    """Trains AutoEncoder."""
    print("Training started for {} epochs.".format(num_epochs))
    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()
        epoch_start = time.time()
        total_examples = 0
        running_loss = 0.0

        iters = 1

        for batch in trainloader:
            iter_time = time.time()

            # zero gradients
            optimizer.zero_grad()

            # forward
            x = batch['data'].to(device)
            out = decoder(encoder(x))

            # calculate loss
            loss = criterion(x, out)

            # backward
            loss.backward()

            # update params
            optimizer.step()

            # track loss
            running_loss = running_loss + loss.item() * batch['data'].size(0)
            total_examples += batch['data'].size(0)
            if print_iter:
                print("Iteration #{}, iter time: {}, loss: {}".format(iters, time.time() - iter_time, loss.item()))
            iters += 1

        if epoch % checkpoint == 0:
            torch.save(encoder.state_dict(), "{}models/{}_encoder_epoch_{}".format(experiment_dir,
                                                                                   model_name,
                                                                                   epoch + start_epoch))
            torch.save(decoder.state_dict(), "{}models/{}_decoder_epoch_{}".format(experiment_dir,
                                                                                   model_name,
                                                                                   epoch + start_epoch))

        epoch_loss = running_loss / total_examples
        print("Epoch #{}/{},  epoch loss: {}, epoch time: {:.5f} seconds".format(epoch + start_epoch,
                                                                                 num_epochs,
                                                                                 epoch_loss,
                                                                                 time.time() - epoch_start))
        # evaluate on trainloader
        if epoch % eval_epoch == 0:
            evaluate_ae(encoder, decoder, criterion, device, trainloader, print_iter=print_iter)

        if scheduler is not None:
            scheduler.step(epoch_loss)
