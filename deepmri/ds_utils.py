import itertools
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
                               orients=(0, 1, 2)):
    """Creates axial, coronal, sagittal volumes.

    Args:
      csv_file: csv file with dMRI paths.
      save_dir: Directory to save the data.
      orients: 0 - Sagittal, 1 - Coronal, 2 - Axial (Default value = (0, 1, 2)

    Returns:
        None
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


def create_multilabel_mask(labels, masks_path, vol_size=(145, 174, 145)):
    """Creates multilabel binary mask.

    Args:
      labels: List of labels, first element is always 'background'
      masks_path: Path to the binary masks.
      vol_size:  Spatial dimensions of volume. (Default value = (145, 174, 145)

    Returns:
        Multi label binary mask.
    """

    mask_ml = np.zeros((*vol_size, len(labels)))
    background = np.ones(vol_size)  # everything that contains no bundle

    # first label must always be the 'background'
    for idx, label in enumerate(labels[1:], 1):
        mask = nib.load(os.path.join(masks_path, label + '_binary_mask.nii.gz'))
        mask_data = mask.get_data()  # dtype: uint8
        mask_ml[:, :, :, idx] = mask_data
        background[mask_data == 1] = 0  # remove this bundle from background

    mask_ml[:, :, :, 0] = background
    return mask_ml.astype('uint8')


def create_voxel_level_dataset(dmri_data, ml_masks, train_coords):
    """Creates voxel level dataset.

    Args:
      dmri_data: numpy.ndarray of shape WxHxDxT: diffusion MRI data.
      ml_masks: numpy.ndarray of shape WxHxDxC: multilabel binary mask volume.
      train_coords: 

    Returns:
      X_train, y_train, X_test, y_test
    """

    # all coord triples
    vs = dmri_data.shape
    all_coords = itertools.product(range(vs[0]),
                                   range(vs[1]),
                                   range(vs[2]))

    # create train set
    train_data = [(dmri_data[crd[0], crd[1], crd[2], :],
                   ml_masks[crd[0], crd[1], crd[2], :])
                  for crd in train_coords]
    x_train, y_train = zip(*train_data)

    # create test set from the remaining points
    test_data = [(dmri_data[crd[0], crd[1], crd[2], :],
                  ml_masks[crd[0], crd[1], crd[2], :])
                 for crd in all_coords if crd not in train_coords]
    x_test, y_test = zip(*test_data)

    return x_train, y_train, x_test, y_test
