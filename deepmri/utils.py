import math
import time
import numpy as np
import torch

from deepmri.vis_utils import visualize_ae_results


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


def dataset_performance(dataset,
                        encoder,
                        decoder,
                        criterion,
                        device,
                        t=0,
                        every_iter=10 ** 10,
                        eval_mode=True,
                        plot=False,
                        mu=None,
                        std=None,
                        scale_back=True,
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
      mu: Mean value.
      std: Standard deviation.
      scale_back: If True loss will be calculated on original voxel values.
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

    best_img = None
    worst_img = None

    print("Total examples: {}".format(total_examples))

    with torch.no_grad():
        running_loss = 0
        eval_start = time.time()
        for i, data in enumerate(dataset, 1):
            x = data['data'].unsqueeze(0).to(device)

            out = decoder(encoder(x))

            # scale back to original voxel values
            if scale_back:
                if (mu is None) and (std is None):
                    x = x.detach().cpu().squeeze()
                    out = out.detach().cpu().squeeze()
                    x = x * data['stds'] + data['means']
                    out = out * data['stds'] + data['means']

                    x = x.unsqueeze(0).to(device)
                    out = out.unsqueeze(0).to(device)
                else:
                    x = x * std + mu
                    out = out * std + mu

                out = out.clamp(min=x.min())

            if masked_loss:
                mask = data['mask'].unsqueeze(1).to(device)
                x = torch.mul(x, mask)
                out = torch.mul(out, mask)
                loss = criterion(x, out) / torch.sum(mask)
            else:
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

        print("Showing slice at t={}".format(t))

        # show best reconstruction
        mu_best, mu_worst = mu, mu
        std_best, std_worst = std, std

        if mu_best is None:
            mu_best = best_img['means'].to(device)
            mu_worst = worst_img['means'].to(device)

        if std_best is None:
            std_best = best_img['stds'].to(device)
            std_worst = worst_img['stds'].to(device)

        visualize_ae_results(best_img['data'],
                             encoder,
                             decoder,
                             criterion,
                             device,
                             mu_best,
                             std_best,
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
                             mu_worst,
                             std_worst,
                             t,
                             suptitle='Worst reconstruction',
                             scale_back=True,
                             cmap='gray')


def evaluate_ae(encoder,
                decoder,
                criterion,
                device,
                trainloader,
                print_iter=False,
                masked_loss=False
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
            x = batch['data'].to(device)
            out = decoder(encoder(x))

            # calculate loss
            if masked_loss:
                mask = batch['mask'].unsqueeze(1).to(device)
                x = torch.mul(x, mask)
                out = torch.mul(out, mask)
                loss = criterion(x, out) / torch.sum(mask)
            else:
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
             eval_epoch=5,
             masked_loss=False
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

    Returns:
        None
    """
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
            if masked_loss:
                mask = batch['mask'].unsqueeze(1).to(device)
                x = torch.mul(x, mask)
                out = torch.mul(out, mask)
                loss = criterion(x, out) / torch.sum(mask)
            else:
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
            evaluate_ae(encoder, decoder, criterion, device, trainloader, print_iter=print_iter,
                        masked_loss=masked_loss)

        if scheduler is not None:
            scheduler.step(epoch_loss)
