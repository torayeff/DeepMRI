import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import numpy as np


class VoxelDataset(Dataset):
    """Voxel dataset for dMRI."""

    def __init__(self,
                 data_dir,
                 file_name,
                 normalize=False,
                 scale=False,
                 noise_prob=None):
        """
        Args:
            data_dir: Data directory with mask and diffusion image.
            file_name: File name.
            normalize: If True, data will be normalized to zero mean and std=1
            scale: If True, data will be scaled by max value.
            noise_prob: If is not None, every element will me zeroed out with probability prob
        """

        print("Loading data...")
        if file_name.endswith('.nii.gz'):
            self.data = nib.load(os.path.join(data_dir, file_name)).get_data()
        else:
            self.data = np.load(os.path.join(data_dir, file_name))['data']
        mask = nib.load(os.path.join(data_dir, 'nodif_brain_mask.nii.gz')).get_data()
        print("Making training set...")
        self.coords = np.transpose(np.nonzero(mask)).tolist()
        self.coords = [
            crd for crd in self.coords if
            len(np.nonzero(self.data[crd[0], crd[1], crd[2]])[0]) != 0
        ]
        self.normalize = normalize
        self.scale = scale
        self.noise_prob = noise_prob

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]

        # make cube
        voxels = self.data[coord[0], coord[1], coord[2], :].reshape(-1)
        voxels = torch.tensor(voxels).float()
        if self.scale:
            voxels = voxels/voxels.max()
        if self.normalize:
            voxels = voxels/voxels.max()
            voxels = (voxels - voxels.mean()) / voxels.std()

        sample = {'data': voxels, 'coord': coord}

        # zero each element with probability self.noise_prob
        if self.noise_prob is not None:
            sample['noisy_data'] = voxels * voxels.bernoulli(1 - self.noise_prob)

        return sample


class OrientationDataset(Dataset):
    """Orientation dataset for dMRI."""

    def __init__(self,
                 data_dir,
                 file_names=None,
                 normalize=True,
                 scale=True,
                 bg_zero=True,
                 sort_fns=True,
                 noise_prob=None):
        """
        Args:
            data_dir: Directory with .npz volumes.
            file_names: File names in dat_dir.
            normalize: If True, the data will be normalized.
            scale: If True, the data will be scaled between 0 and 1.
            bg_zero: If True, background values will be zeroed after normalization.
            sort_fns: If True sorts file_names
            noise_prob: If is not None, every element will me zeroed out with probability noise_prob
        """

        self.data_dir = data_dir
        self.file_names = file_names
        self.scale = scale
        self.normalize = normalize
        self.bg_zero = bg_zero
        self.noise_prob = noise_prob

        if file_names is None:
            self.file_names = os.listdir(data_dir)

        if sort_fns:
            self.file_names = sorted(self.file_names)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.data_dir, file_name)

        orient_img = np.load(file_path)
        x = torch.tensor(orient_img['data']).float()
        mask = torch.tensor(orient_img['mask']).float()
        means = torch.tensor(orient_img['means']).float()[..., None, None]  # add extra dims
        stds = torch.tensor(orient_img['stds']).float()[..., None, None]  # extra dims

        if self.scale:
            x = x/x.max()

        if self.normalize:
            x = (x - means)/stds

        # make background values zero
        if self.bg_zero:
            x[:, mask == 0] = 0

        sample = {'data': x, 'file_name': file_name, 'means': means, 'stds': stds, 'mask': mask}

        # zero each element with probability self.noise_prob
        if self.noise_prob is not None:
            sample['noisy_data'] = x * x.bernoulli(1 - self.noise_prob)

        return sample
