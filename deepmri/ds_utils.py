import os
import time
import nibabel as nib
import numpy as np
import pandas as pd
from dipy.tracking import utils as utils_trk
from scipy import ndimage


def pooled_mean(mu_x, n_x, mu_y, n_y):
    """Calculates pooled mean.

    Args:
      mu_x: mean of group x.
      n_x: number of elements in group x.
      mu_y: mean of group y.
      n_y: 

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
      total_n: 

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
                               orients=(0, 1, 2),
                               th_sum=0,
                               within_brain=True):
    """Creates axial, coronal, sagittal volumes.

    Args:
      csv_file: csv file with dMRI paths.
      save_dir: Directory to save the data.
      orients: 0 - Sagittal, 1 - Coronal, 2 - Axial (Default value = (0, 1, 2)
      th_sum: if volume sum is less than th_sum, do not save it.
      within_brain: If True only regions within brain mask will be considered.

    Returns:
        None
    """
    orient_names = ['sagittal', 'coronal', 'axial']

    df = pd.read_csv(csv_file)
    for _, row in df.iterrows():

        print("Loading file: {}".format(row['dmri_path']))
        data = nib.load(row['dmri_path']).get_fdata()  # 4D data: W x H x D x T
        brain_mask = None

        if within_brain:
            print("Processing only within brain.")
            last_slash = row['dmri_path'].rfind('/')
            bm_path = os.path.join(row['dmri_path'][:last_slash], 'nodif_brain_mask.nii.gz')
            brain_mask = nib.load(bm_path).get_data()

        for orient in orients:

            orient_name = orient_names[orient]
            print("Processing {} orientation...".format(orient_name))
            st = time.time()

            if th_sum == 'avg':
                orient_masks = None
                if orient == 0:
                    orient_masks = [mask for mask in brain_mask]
                elif orient == 1:
                    orient_masks = [mask for mask in brain_mask.transpose(1, 0, 2)]
                elif orient == 2:
                    orient_masks = [mask for mask in brain_mask.transpose(2, 0, 1)]

                th_sum = sum(orient_masks).sum() / data.shape[orient]
                print("Threshold for {} is {}".format(orient_names[orient], th_sum))

            for idx in range(data.shape[orient]):
                orient_img = None
                slc_mask = None
                roi_idxs = None

                if orient == 0:
                    orient_img = data[idx, :, :, :]
                elif orient == 1:
                    orient_img = data[:, idx, :, :]
                elif orient == 2:
                    orient_img = data[:, :, idx, :]

                if within_brain:
                    if orient == 0:
                        slc_mask = brain_mask[idx, :, :]
                    elif orient == 1:
                        slc_mask = brain_mask[:, idx, :]
                    elif orient == 2:
                        slc_mask = brain_mask[:, :, idx]

                    roi_idxs = np.nonzero(slc_mask)
                    check_sum = np.sum(slc_mask)
                else:
                    check_sum = np.sum(orient_img)

                if check_sum <= th_sum:
                    print("{} idx={} is empty. Skipping.".format(orient_name, idx))
                else:
                    save_path = "{}/data_{}_{}_idx_{:03d}".format(orient_name, row['subj_id'], orient_name, idx)
                    print("Saving: {}".format(save_path))
                    save_path = os.path.join(save_dir, save_path)
                    orient_img = orient_img.transpose(2, 0, 1)  # channel x width x height)

                    means = []
                    stds = []

                    for ch in range(orient_img.shape[0]):

                        if within_brain:
                            # calculate stats only within brain region
                            roi = orient_img[ch][roi_idxs]
                            mu = roi.mean()
                            std = roi.std()
                        else:
                            mu = orient_img[ch].mean()
                            std = orient_img[ch].std()

                        means.append(mu)
                        stds.append(std)

                    np.savez(save_path,
                             data=orient_img,
                             mask=slc_mask.astype('uint8'),
                             means=np.array(means),
                             stds=np.array(stds))

            print("Processed in {:.5f} seconds.".format(time.time() - st))
    print("Done!")


def get_number_of_points(strmlines):
    """Adapted from https://github.com/MIC-DKFZ/TractSeg/issues/39#issuecomment-496181262

    Args:
      strmlines: nibabel streamlines
    Returns:
        Number of point in streamlines.
    """
    count = 0
    for sl in strmlines:
        count += len(sl)
    return count


def remove_small_blobs(img, threshold=1):
    """Adapted from https://github.com/MIC-DKFZ/TractSeg/issues/39#issuecomment-496181262
    Find blobs/clusters of same label. Only keep blobs with more than threshold elements.
    This can be used for postprocessing.

    Args:
      img: Threshold.
      threshold:  (Default value = 1)

    Returns:
        Binary mask.
    """
    # mask, number_of_blobs = ndimage.label(img, structure=np.ones((3, 3, 3)))  #Also considers diagonal elements for
    # determining if a element belongs to a blob -> not so good, because leaves hardly any small blobs we can remove
    mask, number_of_blobs = ndimage.label(img)
    print('Number of blobs before filtering: ' + str(number_of_blobs))
    counts = np.bincount(mask.flatten())  # number of pixels in each blob
    print(counts)

    remove = counts <= threshold
    remove_idx = np.nonzero(remove)[0]

    for idx in remove_idx:
        mask[mask == idx] = 0  # set blobs we remove to 0
    mask[mask > 0] = 1  # set everything else to 1

    mask_after, number_of_blobs_after = ndimage.label(mask)
    print('Number of blobs after filtering: ' + str(number_of_blobs_after))

    return mask


def create_tract_mask(trk_file_path, mask_output_path, ref_img_path, hole_closing=0, blob_th=10):
    """Adapted from https://github.com/MIC-DKFZ/TractSeg/issues/39#issuecomment-496181262
    Creates binary mask from streamlines in .trk file.

    Args:
      trk_file_path: Path for the .trk file
      mask_output_path: Path to save the binary mask.
      ref_img_path: Path to the reference image to get affine and shape
      hole_closing: Integer for closing the holes. (Default value = 0)
      blob_th: Threshold for removing small blobs. (Default value = 10)

    Returns:
        None
    """

    ref_img = nib.load(ref_img_path)
    ref_affine = ref_img.affine
    ref_shape = ref_img.shape

    streamlines = nib.streamlines.load(trk_file_path).streamlines

    # Upsample Streamlines  (very important, especially when using DensityMap Threshold. Without upsampling eroded
    # results)
    print("Upsampling...")
    print("Nr of points before upsampling " + str(get_number_of_points(streamlines)))
    max_seq_len = abs(ref_affine[0, 0] / 4)
    print("max_seq_len: {}".format(max_seq_len))
    streamlines = list(utils_trk.subsegment(streamlines, max_seq_len))
    print("Nr of points after upsampling " + str(get_number_of_points(streamlines)))

    # Remember: Does not count if a fibers has no node inside of a voxel -> upsampling helps, but not perfect
    # Counts the number of unique streamlines that pass through each voxel -> oversampling does not distort result
    dm = utils_trk.density_map(streamlines, ref_shape, affine=ref_affine)

    # Create Binary Map
    dm_binary = dm > 1  # Using higher Threshold problematic, because often very sparse
    dm_binary_c = dm_binary

    # Filter Blobs
    dm_binary_c = remove_small_blobs(dm_binary_c, threshold=blob_th)

    # Closing of Holes (not ideal because tends to remove valid holes, e.g. in MCP)
    if hole_closing > 0:
        size = hole_closing
        dm_binary_c = ndimage.binary_closing(dm_binary_c, structure=np.ones((size, size, size))).astype(dm_binary.dtype)

    # Save Binary Mask
    dm_binary_img = nib.Nifti1Image(dm_binary_c.astype("uint8"), ref_affine)
    nib.save(dm_binary_img, mask_output_path)


def create_multilabel_mask(labels, masks_path, nodif_brain_mask_path, vol_size=(145, 174, 145)):
    """Creates multilabel binary mask.

    Args:
      labels: List of labels, first element is always 'background', second element is always 'other'.
      masks_path: Path to the binary masks.
      nodif_brain_mask_path: Path no diffusion brain mask.
      vol_size:  Spatial dimensions of volume. (Default value = (145, 174, 145)

    Returns:
        Multi label binary mask.
    """

    mask_ml = np.zeros((*vol_size, len(labels)))
    background = np.ones(vol_size)  # everything that contains no bundle
    other = nib.load(nodif_brain_mask_path).get_data().astype('uint8')
    background[other == 1] = 0  # what is within brain is not background ?
    mask_ml[:, :, :, 0] = background

    # first label must always be the 'background'
    for idx, label in enumerate(labels[2:], 2):
        mask = nib.load(os.path.join(masks_path, label + '_binary_mask.nii.gz'))
        mask_data = mask.get_data()  # dtype: uint8
        mask_ml[:, :, :, idx] = mask_data
        other[mask_data == 1] = 0  # remove this from other class

    mask_ml[:, :, :, 1] = other

    return mask_ml.astype('uint8')


def create_data_masks(ml_masks, slice_orients, labels, max_samples_per_label):
    """Creates multilabel binary mask for training from full multilabel binary mask.
    Args:
        ml_masks: Full multilabel binary mask.
        slice_orients: Slice orientations and their indices to take training voxels,
                        e.g., slice_orients = [('sagittal', 72), ('sagittal', 89)]
        labels: Labels.
        max_samples_per_label: Maximum number of samples per label to take for training.
    Returns:
        Multilabe binary mask for training.
    """

    data_masks = np.zeros(ml_masks.shape)

    for orient in slice_orients:
        if orient[0] == 'sagittal':
            slc = ml_masks[orient[1], :, :]
            data_slc = data_masks[orient[1], :, :]
        elif orient[0] == 'coronal':
            slc = ml_masks[:, orient[1], :]
            data_slc = data_masks[:, orient[1], :]
        elif orient[0] == 'axial':
            slc = ml_masks[:, :, orient[1]]
            data_slc = data_masks[:, :, orient[1]]
        else:
            print('Invalid orientation name was given.')
            continue

        for ch, label in enumerate(labels):
            max_samples = max_samples_per_label[ch]
            label_mask = slc[:, :, ch]

            label_coords = np.nonzero(label_mask)
            clen = len(label_coords[0])

            # if the number of annotations is more than max_samples
            # take max_sample annotations from the middle, and zero out others
            # else take all the annotations
            if clen > max_samples:
                start = (clen // 2) - (max_samples // 2) - (max_samples % 2)
                end = start + max_samples
                annot_coords = (label_coords[0][start:end], label_coords[1][start:end])
            else:
                annot_coords = label_coords

            # assign annotations
            data_slc[:, :, ch][annot_coords] = 1

    total = 0
    for ch, label in enumerate(labels):
        annots = len(np.nonzero(data_masks[:, :, :, ch])[0])
        total += annots
        print("\"{}\" has {} annotations.".format(labels[ch], annots))
    print("Total annotations: {}".format(total))

    return data_masks


def create_dataset_from_data_mask(features, data_masks, labels=None, multi_label=False):
    """Creates voxel level dataset.

        Args:
          features: numpy.ndarray of shape WxHxDxT: diffusion MRI data.
          data_masks: numpy.ndarray of shape WxHxDxC: multilabel binary mask volume.
          labels: If not None, names for classes will be used instead of numbers.
          multi_label: If True multi labe one hot encoding labels will be generated.

        Returns:
          x_train, y_train
        """

    x_set, y_set, voxel_coords = [], [], []
    if multi_label:
        voxel_coords = np.nonzero(data_masks)[:3]  # take only spatial dims
        voxel_coords = list(zip(*voxel_coords))  # make triples
        voxel_coords = list(set(voxel_coords))  # remove duplicates
        for pt in voxel_coords:
            x = features[pt[0], pt[1], pt[2], :]
            y = data_masks[pt[0], pt[1], pt[2], :]
            x_set.append(x)
            y_set.append(y)
    else:
        for pt in np.transpose(np.nonzero(data_masks)):
            voxel_coords.append((pt[0], pt[1], pt[2]))
            x = features[pt[0], pt[1], pt[2], :]
            y = pt[3]
            x_set.append(x)

            if labels is None:
                y_set.append(y)
            else:
                y_set.append(labels[y])

    return np.array(x_set), np.array(y_set), np.array(voxel_coords)


def preds_to_data_mask(preds, voxel_coords, labels, vol_size=(145, 174, 145)):
    """Creates data mask from predictions.

    Args:
        preds: Predictions.
        voxel_coords: Coordinates for each prediction.
        labels: Labels.
        vol_size:  Spatial dimensions of volume. (Default value = (145, 174, 145)

    Return:
        Binary mask.
    """
    data_mask = np.zeros((*vol_size, len(labels)))

    for pred, crd in zip(preds, voxel_coords):
        for ch, v in enumerate(pred):
            if v != 0:
                data_mask[crd[0], crd[1], crd[2], ch] = 1

    return data_mask


def features_from_coords(features, coords, orient, scale=1):
    orient_features = []
    for crd in coords:
        if orient == 'sagittal':
            orient_features.append(features[crd[0], crd[1] // scale, crd[2] // scale, :])
        elif orient == 'coronal':
            orient_features.append(features[crd[1], crd[0] // scale, crd[2] // scale, :])
        elif orient == 'axial':
            orient_features.append(features[crd[2], crd[0] // scale, crd[1] // scale, :])
        else:
            raise ValueError('Unknown orientation.')

    return np.array(orient_features)
