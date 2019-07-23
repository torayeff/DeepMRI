import math
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from deepmri.CustomLosses import MaskedLoss
from deepmri import visutils
from matplotlib.lines import Line2D


def calc_conv_dim(w, k, s, p):
    """Calculates output dimensions of convolution operator.

    Args:
      w: width
      k: kernel size
      s: stride
      p: padding

    Returns:
        None
    """

    dim = ((w + 2 * p - k) / s) + 1
    print("Conv dim: ", dim, math.floor(dim))


def calc_transpose_conv_dim(w, k, s, p, out_p):
    """Calculates output dimensions of transpose convolution operator.

    Args:
      w: width
      k: kernel
      s: strid
      p: padding
      out_p: out padding

    Returns:
        None
    """

    dim = (w - 1) * s - 2 * p + k + out_p
    print("Deconv dim: ", dim, math.floor(dim))


def pad_3d(img_to_pad, target_dims=(256, 256, 256)):
    """Pads given 3D image.

    Args:
      img_to_pad: Image to pad
      target_dims:  Target dimensions. (Default value = (256, 256, 256):

    Returns:
        None
    """

    diffs = np.array(target_dims) - np.array(img_to_pad.shape)
    pads = tuple([(d // 2 + (d % 2), d // 2) for d in diffs])
    return np.pad(img_to_pad, pads, 'constant')


def count_model_parameters(model):
    """Counts total parameters of model.

    Args:
      model: Pytorch model

    Returns:
        The number of total and trainable parameters.
    """

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


def evaluate_adhd_classifier(classifier, rnn_encoder, criterion, dataloader, device):
    """Evaluate ADHD Classifier.

    Args:
      classifier: Classifier model.
      rnn_encoder: RNN Encoder model.
      criterion: Loss function.
      dataloader: Dataloader.
      device: device.

    Returns:
        All labels and all predictions.
    """

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
    """Evaluates RNN AE.

    Args:
      encoder: Encoder model.
      decoder: Decoder model.
      criterion: Criterion.
      dataloader: Data loader.
      device: device.

    Returns:
        None
    """
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


def img_stats(sample, t=None):
    mask = sample['mask']
    x = sample['data']
    y = sample['out']

    # scale_back
    x_back = x * sample['stds'] + sample['means']
    y_back = y * sample['stds'] + sample['means']

    # clamp negative and over maximum values
    y_back = y_back.clamp(min=x_back.min(), max=x_back.max())

    # zero out background
    x[:, mask == 0] = 0
    y[:, mask == 0] = 0
    x_back[:, mask == 0] = 0
    y_back[:, mask == 0] = 0

    masked_mse = MaskedLoss()
    mse = nn.MSELoss(reduction='mean')

    loss = masked_mse(x.unsqueeze(0), y.unsqueeze(0), mask.unsqueeze(0).unsqueeze(0))
    scaled_loss = masked_mse(x_back.unsqueeze(0),
                             y_back.unsqueeze(0),
                             mask.unsqueeze(0).unsqueeze(0))

    print("Avg. loss: {:.5f}, Avg. scaled loss: {:.5f}".format(loss.item(), scaled_loss.item()))

    if t is not None:
        slc_x = x_back[t]
        slc_y = y_back[t]
        roi_x = slc_x[mask == 1]
        roi_y = slc_y[mask == 1]
        roi_loss = mse(roi_x, roi_y)
        min_x, max_x = slc_x.min().item(), slc_x.max().item()
        min_y, max_y = slc_y.min().item(), slc_y.max().item()

        visutils.show_slices(
            [slc_x.numpy(), slc_y.numpy(), mask.numpy()],
            titles=['x, min: {:.2f}, max: {:.2f}'.format(min_x, max_x),
                    'y, min: {:.2f}, max: {:.2f}'.format(min_y, max_y),
                    'mask'],
            suptitle='Loss in ROI: {:.2f}'.format(roi_loss.item()),
            cmap='gray'
        )


def dataset_performance(dataset,
                        encoder,
                        decoder,
                        criterion,
                        device,
                        t=0,
                        every_iter=10 ** 10,
                        eval_mode=True,
                        plot=False,
                        masked_loss=False):
    """Calculates average loss on whole dataset.

    Args:
      dataset: Dataset
      encoder: Encoder model.
      decoder: Decoder model.
      criterion: Criterion.
      device: device.
      t: time index for plotting.
      every_iter:  Print statistics every iteration. (Default value = 10 ** 10)
      eval_mode:  Boolean for the model mode. (Default value = True)
      plot:  Boolean to plot. (Default value = False)
      masked_loss: If True loss will be calculated over masked region only.

    Returns:
        None
    """

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

    best = None
    worst = None

    print("Total examples: {}".format(total_examples))

    with torch.no_grad():
        running_loss = 0
        eval_start = time.time()
        for i, data in enumerate(dataset, 1):
            x = data['data'].unsqueeze(0).to(device)
            feature = encoder(x)
            out = decoder(feature)
            mask = data['mask'].unsqueeze(0).unsqueeze(0).to(device)

            if masked_loss:
                loss = criterion(x, out, mask)
            else:
                loss = criterion(x, out)

            if loss.item() < min_loss:
                min_loss = loss.item()
                best = data
                best['out'] = out.detach().cpu().squeeze()
                best['feature'] = feature.detach().cpu().squeeze()

            if loss.item() > max_loss:
                max_loss = loss.item()
                worst = data
                worst['out'] = out.detach().cpu().squeeze()
                worst['feature'] = feature.detach().cpu().squeeze()

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
                  best['file_name'],
                  worst['file_name']))

    return best, worst


def evaluate_ae(encoder,
                decoder,
                criterion,
                device,
                trainloader,
                print_iter=False,
                masked_loss=False,
                denoising=False
                ):
    """Evaluates AE.

    Args:
      encoder: Encoder model.
      decoder: Decoder model.
      criterion: Criterion.
      device: Device
      trainloader: Train loader
      print_iter:  Print every iteration. (Default value = False)
      masked_loss: If True loss will be calculated over masked region only.
      denoising: If True, denoising AE will be evaluated.

    Returns:
        Average loss.
    """

    start = time.time()
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        total_examples = 0
        running_loss = 0.0

        for i, batch in enumerate(trainloader, 1):

            # forward
            x = batch['data'].to(device)
            if denoising:
                h = encoder(batch['noisy_data'].to(device))
            else:
                h = encoder(x)
            y = decoder(h)

            # calculate loss
            if masked_loss:
                mask = batch['mask'].unsqueeze(1).to(device)
                loss = criterion(y, x, mask)
            else:
                loss = criterion(y, x)

            # track loss
            running_loss = running_loss + loss.item() * batch['data'].size(0)
            total_examples += batch['data'].size(0)

            if print_iter:
                print("Batch #{}, Batch loss: {:.5f}".format(i, loss.item()))

        avg_loss = running_loss / total_examples

    print("Evaluated {} examples, Avg. loss: {:.5f}, Time: {:.5f}".format(total_examples, avg_loss,
                                                                          time.time() - start))
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
             eval_epoch=5,
             masked_loss=False,
             sparsity=None,
             denoising=False,
             prec=5
             ):
    """Trains AutoEncoder.

    Args:
      encoder: Encoder model.
      decoder: Decoder model.
      criterion: Criterion.
      optimizer: Optimizer.
      device: Device
      trainloader: Trainloader.
      num_epochs: Number of epochs to train.
      model_name: Model name for saving.
      experiment_dir: Experiment directory with data.
      start_epoch:  Starting epoch. Useful for resuming.(Default value = 0)
      scheduler:  Learning rate scheduler.(Default value = None)
      checkpoint:  Save every checkpoint epoch. (Default value = 1)
      print_iter:  Print every iteration. (Default value = False)
      eval_epoch:  Evaluate every eval_epoch epoch. (Default value = 5)
      masked_loss: If True loss will be calculated over masked region only.
      sparsity: If not None, sparsity penalty will be applied to hidden activations.
      denoising: If True, Denoising AE will be trained.
      prec: Error precision.

    Returns:
        None
    """
    print("Training started for {} epochs.".format(num_epochs))
    if sparsity is not None:
        print('Sparsity is on with lambda={}'.format(sparsity))
    if denoising:
        print('Training denoising autencoder.')

    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()
        epoch_start = time.time()
        total_examples = 0
        running_loss = 0.0

        iters = 1

        for batch in trainloader:
            iter_time = time.time()

            # forward
            x = batch['data'].to(device)
            if denoising:
                h = encoder(batch['noisy_data'].to(device))
            else:
                h = encoder(x)
            y = decoder(h)

            # calculate loss
            if masked_loss:
                mask = batch['mask'].unsqueeze(1).to(device)
                loss = criterion(y, x, mask)
            else:
                loss = criterion(y, x)

            if sparsity is not None:
                loss = loss + sparsity * torch.abs(h).sum()

            # zero gradients
            optimizer.zero_grad()

            # backward
            loss.backward()

            # update params
            optimizer.step()

            # track loss
            running_loss = running_loss + loss.item() * batch['data'].size(0)
            total_examples += batch['data'].size(0)
            if print_iter:
                print("Iteration #{}, loss: {:.{}f}, iter time: {}".format(iters,
                                                                           loss.item(),
                                                                           prec,
                                                                           time.time() - iter_time))
            iters += 1

        if epoch % checkpoint == 0:
            torch.save(encoder.state_dict(), "{}saved_models/{}_encoder_epoch_{}".format(experiment_dir,
                                                                                         model_name,
                                                                                         epoch + start_epoch))
            torch.save(decoder.state_dict(), "{}saved_models/{}_decoder_epoch_{}".format(experiment_dir,
                                                                                         model_name,
                                                                                         epoch + start_epoch))

        epoch_loss = running_loss / total_examples
        print("Epoch #{}/{},  epoch loss: {:.{}f}, epoch time: {:.5f} seconds".format(epoch + start_epoch,
                                                                                      num_epochs + start_epoch,
                                                                                      epoch_loss,
                                                                                      prec,
                                                                                      time.time() - epoch_start))
        # evaluate on trainloader
        if epoch % eval_epoch == 0:
            evaluate_ae(encoder, decoder, criterion, device, trainloader, print_iter=print_iter,
                        masked_loss=masked_loss, denoising=denoising)

        if scheduler is not None:
            scheduler.step(epoch_loss)


def plot_grad_flow(named_parameters):
    """
    Credits https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
