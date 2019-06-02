import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def show_slices(slices,
                suptitle="Visualization",
                titles=('Saggital', 'Coronal', 'Axial'),
                figsize=(10, 5),
                fontsize=24,
                cmap=None):
    """Function to display row of image slices

    Args:
      slices: Slices to show
      suptitle:  (Default value = "Visualization")
      titles:  (Default value = ('Saggital', 'Coronal', 'Axial'):
      figsize:  (Default value = (10, 5):
      fontsize:  (Default value = 24)
      cmap:  (Default value = None)

    Returns:
        None
    """

    plt.rcParams["figure.figsize"] = figsize
    fig, axes = plt.subplots(1, len(slices))
    fig.suptitle(suptitle, fontsize=fontsize)
    for i, slc in enumerate(slices):
        axes[i].set_title(titles[i])
        axes[i].imshow(slc.T, cmap=cmap, origin="lower")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()


def show_masked_slices(slices,
                       masks,
                       suptitle="Visualization",
                       titles=('Saggital', 'Coronal', 'Axial'),
                       figsize=(10, 5),
                       fontsize=24,
                       cmap=matplotlib.cm.gray,
                       mask_color='red',
                       alpha=0.9):
    """Function to display row of image slices

    Args:
      slices: Slices to show
      masks: Masks.
      suptitle:  Sup title. (Default value = "Visualization")
      titles:  (Default value = ('Saggital', 'Coronal', 'Axial'):
      figsize:  (Default value = (10, 5):
      fontsize:  (Default value = 24)
      cmap:  (Default value = matplotlib.cm.gray)
      mask_color:  Color for mask. (Default value = 'red')
      alpha:  Opacity. (Default value = 0.9)

    Returns:
        None
    """

    plt.rcParams["figure.figsize"] = figsize
    fig, axes = plt.subplots(1, len(slices))
    fig.suptitle(suptitle, fontsize=fontsize)
    for i, slc in enumerate(slices):
        masked_img = np.ma.array(slc.T, mask=masks[i].T)
        cmap.set_bad(mask_color, alpha=alpha)

        axes[i].set_title(titles[i])
        axes[i].imshow(masked_img, cmap=cmap, origin="lower")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()


def show_one_slice(slc,
                   title="One Slice",
                   figsize=(10, 5),
                   fontsize=12,
                   cmap=None):
    """

    Args:
      slc: 
      title:  (Default value = "One Slice")
      figsize:  (Default value = (10)
      5): 
      fontsize:  (Default value = 12)
      cmap:  (Default value = None)

    Returns:

    """
    plt.rcParams["figure.figsize"] = figsize
    plt.imshow(slc.T, cmap=cmap, origin="lower")
    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def show_one_masked_slice(img,
                          mask,
                          cmap=matplotlib.cm.gray,
                          title="Masked Image",
                          figsize=(10, 5),
                          fontsize=12,
                          mask_color='red',
                          alpha=0.9):
    """

    Args:
      img: 
      mask: 
      cmap:  (Default value = matplotlib.cm.gray)
      title:  (Default value = "Masked Image")
      figsize:  (Default value = (10)
      5): 
      fontsize:  (Default value = 12)
      mask_color:  (Default value = 'red')
      alpha:  (Default value = 0.9)

    Returns:

    """
    masked_img = np.ma.array(img, mask=mask)
    cmap.set_bad(mask_color, alpha=alpha)
    plt.rcParams["figure.figsize"] = figsize
    plt.imshow(masked_img, origin='lower', cmap=cmap)
    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Credits: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    Args:
      cm: 
      classes: 
      normalize:  (Default value = False)
      title:  (Default value = 'Confusion matrix')
      cmap:  (Default value = plt.cm.Blues)

    Returns:

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


def visualize_ae_results(x, encoder, decoder, criterion, device, mu, std, t,
                         scale_back=True,
                         suptitle="Visualization",
                         cmap=None):
    """Visualizes AE results.

    Args:
      x: Data to to evaluate and visualize.
      encoder: Encoder model.
      decoder: Decoder model.
      criterion: Criterion.
      device: Device.
      mu: Mean value.
      std: Standard deviation.
      t: Time point.
      scale_back:  If True scales back to the original voxel values. (Default value = True)
      suptitle:  Sup title.(Default value = "Visualization")
      cmap:  Color map. (Default value = None)

    Returns:
        None
    """
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

    original_title = "Original\n Minval: {:.5f}, Maxval: {:.5f}".format(x[t].min(), x[t].max())
    recons_title = "Reconstruction\n Minval: {:.5f}, Maxval: {:.5f}".format(y[t].min(), y[t].max())
    show_slices([
        x[t],
        y[t]
    ], suptitle=suptitle, titles=[original_title, recons_title], fontsize=16, cmap=cmap)
    print("Loss: {}".format(loss_before_scaling))
    if scale_back:
        print("Loss after scaling back: {}".format(loss_after_scaling))
    print("-" * 100)


def visualize_masks(dmri_data,
                    ml_masks_list,
                    labels,
                    x_coord,
                    y_coord,
                    z_coord,
                    t,
                    suptitles,
                    mask_color='red'
                    ):
    """Visualize masks.
    Args:
        dmri_data: Diffusion MRI data
        ml_masks_list: List of masks to visualize.
        labels: Names of labels in masks.
        x_coord: x coordinate.
        y_coord: y coordinate.
        z_coord: z coordinate.
        t: time coordinate.
        suptitles: Suptitles.
        mask_color: Mask color.

    Returns:
        None
    """
    # show binary masks
    slices = [
        dmri_data[x_coord, :, :, t],
        dmri_data[:, y_coord, :, t],
        dmri_data[:, :, z_coord, t]
    ]

    for ch in range(len(labels)):
        for idx, ml_masks in enumerate(ml_masks_list):
            masks = [
                ml_masks[x_coord, :, :, ch],
                ml_masks[:, y_coord, :, ch],
                ml_masks[:, :, z_coord, ch]
            ]

            show_masked_slices(slices,
                               masks,
                               suptitle=suptitles[idx] + labels[ch],
                               mask_color=mask_color)