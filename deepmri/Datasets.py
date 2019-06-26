import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import pickle
import pandas as pd
import h5py
import numpy as np


class SHORESlices(Dataset):
    """SHORE Slices dataset for dMRI."""

    def __init__(self, data_dir, file_name, normalize=False, bg_zero=True):
        """
        Args:
            data_dir: Data directory with mask and diffusion image.
        """

        print("Loading data...")
        if file_name.endswith('.nii.gz'):
            self.data = nib.load(os.path.join(data_dir, file_name)).get_data()
        else:
            self.data = np.load(os.path.join(data_dir, file_name))['data']
        self.mask = nib.load(os.path.join(data_dir, 'nodif_brain_mask.nii.gz')).get_data()
        self.normalize = normalize
        self.indices = []
        self.bg_zero = bg_zero
        for idx in range(174):
            check_sum = np.sum(self.mask[:, idx, :])
            if check_sum > 0:
                self.indices.append(idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        slice_idx = self.indices[idx]
        x = self.data[:, slice_idx, :, :].transpose(2, 0, 1)
        x = torch.tensor(x).float()
        mask = self.mask[:, slice_idx, :]
        mask = torch.tensor(mask).float()

        if self.normalize:
            x = (x - x.mean()) / x.std()

        if self.bg_zero:
            x[:, mask == 0] = 0
        return {'data': x, 'slice_idx': slice_idx, 'mask': mask}


class VoxelDataset(Dataset):
    """Voxel dataset for dMRI."""

    def __init__(self, data_dir, file_name, normalize=False):
        """
        Args:
            data_dir: Data directory with mask and diffusion image.
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

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]

        # make cube
        voxels = self.data[coord[0], coord[1], coord[2], :].reshape(-1)
        voxels = torch.tensor(voxels).float()
        if self.normalize:
            voxels = (voxels - voxels.mean()) / voxels.std()
        return {'data': voxels, 'coord': coord}


class NeighborhoodDataset(Dataset):
    """Neighborhood dataset for dMRI."""

    def __init__(self, data_dir, file_name, nh=3, normalize=False):
        """
        Args:
            data_dir: Data directory with mask and diffusion image.
            nh: Neighborhood.
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
        self.nh = nh
        self.normalize = normalize

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        bx = self.get_borders(coord[0], self.data.shape[0], nh=self.nh)
        by = self.get_borders(coord[1], self.data.shape[1], nh=self.nh)
        bz = self.get_borders(coord[2], self.data.shape[2], nh=self.nh)

        # make cube
        cube = self.data[bx[0]:bx[1], by[0]:by[1], bz[0]:bz[1], :]
        cube = cube.transpose(3, 0, 1, 2)  # transpose to make channels first
        cube = torch.tensor(cube).float()
        if self.normalize:
            cube = (cube - cube.mean()) / cube.std()
        return {'data': cube, 'coord': coord}

    @staticmethod
    def get_borders(crd, border, nh=3):
        start = max(crd - (nh // 2), 0)
        end = min(crd + (nh // 2) + 1, border)

        return start, end


class OrientationDatasetChannelNorm(Dataset):
    """Orientation dataset for dMRI."""

    def __init__(self,
                 data_dir,
                 file_names=None,
                 normalize=True,
                 bg_zero=True,
                 sort_fns=True):
        """
        Args:
            data_dir: Directory with .npz volumes.
            file_names: File names in dat_dir.
            normalize: If True, the data will be normalized.
            bg_zero: If True, background values will be zeroed after normalization.
            sort_fns: If True sorts file_names
        """

        self.data_dir = data_dir
        self.file_names = file_names
        self.normalize = normalize
        self.bg_zero = bg_zero

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

        if self.normalize:
            x = (x - means)/stds

        # make background values zero
        if self.bg_zero:
            x[:, mask == 0] = 0

        sample = {'data': x, 'file_name': file_name, 'means': means, 'stds': stds, 'mask': mask}

        return sample


class OrientationDataset(Dataset):
    """Orientation dataset for dMRI."""

    def __init__(self, data_dir, file_names=None, normalize=True,
                 mu=None, std=None, scale_range=None, to_tensor=True, sort_fns=False):
        """
        Args:
            data_dir: Directory with .npz volumes.
            file_names: File names in dat_dir.
            normalize: If True, the data will be normalized
            mu: Mean of the dataset.
            std: Standard deviation of the dataset.
            scale_range: If not None, data will be scaled into the given range.
            to_tensor: If True, numpy array will be converted to tensor.
            sort_fns: If True sorts file_names
        """

        self.data_dir = data_dir
        self.file_names = file_names
        self.normalize = normalize
        self.mu = mu
        self.std = std
        self.scale_range = scale_range
        self.to_tensor = to_tensor

        if file_names is None:
            self.file_names = os.listdir(data_dir)

        if sort_fns:
            self.file_names = sorted(self.file_names)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.data_dir, file_name)

        x = np.load(file_path)['data']

        if self.normalize:
            if self.mu is None:
                self.mu = x.mean()

            if self.std is None:
                self.std = x.std()

            # mean subtraction
            x = x - self.mu

            # scaling
            if self.scale_range is None:
                x /= self.std
            else:
                a = self.scale_range[0]
                b = self.scale_range[1]
                x = a + ((x - x.min()) * (b - a)) / (x.max() - x.min())

        if self.to_tensor:
            x = torch.tensor(x).float()

        sample = {'data': x, 'file_name': file_name}

        return sample


class Volume3dDataset(Dataset):
    """3D Volume dataset with .3dtensor extension."""

    def __init__(self, root_dir, normalize=True, mu=None, std=None, scale_range=None, sort=False):
        """
        Args:
            root_dir: Directory with all 3d volumes.
            normalize: If true, the data will be normalized.
            mu: Mean for normalization.
            std: Standard deviation for normalization.
            scale_range: tuple, If not None will be scaled in range (a, b), else standardize.
            sort: If True sorts file_paths, useful for debugging.
        """
        self.file_paths = []
        self.mu = mu
        self.std = std
        self.normalize = normalize
        self.scale_range = scale_range

        for file_name in os.listdir(root_dir):
            if file_name.endswith('.3dtensor'):
                self.file_paths.append(os.path.join(root_dir, file_name))

        if sort:
            self.file_paths = sorted(self.file_paths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Fetches 3d volumes."""

        with open(self.file_paths[idx], "rb") as f:
            x = pickle.load(f)  # channel=1 x w x h x

        # mean subtraction and normalization
        if self.normalize:
            if self.mu is None:
                self.mu = x.mean()

            if self.std is None:
                self.std = x.std()

            # mean subtraction
            x = x - self.mu

            # normalization
            if self.scale_range is None:
                x /= self.std
            else:
                a = self.scale_range[0]
                b = self.scale_range[1]
                x = a + ((x - x.min()) * (b - a)) / (x.max() - x.min())

        return x


class Feature4dDataset(Dataset):
    """4D feature dataset."""

    def __init__(self, root_dir, max_seq_len=None):
        """
        Args:
            root_dir: Directory with all 4d features images.
        """
        self.file_paths = []
        for file_name in os.listdir(root_dir):
            if file_name.endswith('.4dtensor'):
                self.file_paths.append(os.path.join(root_dir, file_name))

        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Fetches feature tensors."""

        with open(self.file_paths[idx], "rb") as f:
            x = pickle.load(f)  # time x channel x w x h x d

        if self.max_seq_len:
            return x[:self.max_seq_len]

        return x


class MRIDataset(Dataset):
    """MRI dataset."""

    def __init__(self, root_dir, mu=None, std=None, normalize=True, seq_idxs=(None, None)):
        """
        Args:
            root_dir: Directory with all MRI images.
            mu: mean for the normalization.
            std: standard deviation for the normalization.
            normalize: whether to normalize data or not.
            seq_idxs: begin and start indexes to select for time index.
        """
        self.file_paths = []
        for file_name in os.listdir(root_dir):
            if file_name.endswith('.nii.gz'):
                self.file_paths.append(os.path.join(root_dir, file_name))

        self.mu = mu
        self.std = std
        self.normalize = normalize
        self.seq_idxs = seq_idxs

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Fetches MRI images."""

        x = nib.load(self.file_paths[idx]).get_fdata()

        if len(x.shape) == 4:
            x = x.transpose(3, 0, 1, 2)  # time x width x height x depth
            x = torch.tensor(x).float()[self.seq_idxs[0]:self.seq_idxs[1]]
            x = x.unsqueeze(1)  # Sequence of 3D Volumes: time x channel x width x height x depth
        else:
            x = torch.tensor(x).float()
            x = x.unsqueeze(0)  # 3D Volume: channel x width x height x depth

        if self.normalize:
            if self.mu is None:
                x = (x - x.mean()) / x.std()
            else:
                x = (x - self.mu) / self.std

        return x


class HDF5Dataset(Dataset):
    """HDF5 Dataset with MRI volumes."""

    def __init__(self, file_path, mu=None, std=None):
        super().__init__()

        self.mu = mu
        self.std = std

        self.archive = h5py.File(file_path, 'r')
        self.data = self.archive['data']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x = self.data[str(idx)]

        x = np.array(x)
        x = torch.from_numpy(x).float()

        if (self.mu is not None) and (self.std is not None):
            x = (x - self.mu) / self.std

        return x

    def close(self):
        self.archive.close()


class ADHDFeatureDataset(Dataset):
    """ADHD features dataset."""

    def __init__(self, root_dir, csv_file, seq_len=None, binary=True, sep=','):
        self.file_names = []
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.seq_len = seq_len
        self.binary = binary

        self.df = pd.read_csv(csv_file, sep=sep)

        for file_name in os.listdir(root_dir):
            if file_name.endswith('.4dtensor') and ((file_name[:-9] + '.nii.gz') in self.df['fmri'].values):
                self.file_names.append(file_name)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        with open(os.path.join(self.root_dir, self.file_names[idx]), "rb") as f:
            x = pickle.load(f)  # time x channel x w x h x d

        if self.seq_len:
            x = x[:self.seq_len]

        fk = self.file_names[idx][:-9] + ".nii.gz"

        y = int(self.df[self.df['fmri'] == fk]['dx'].item())

        if self.binary and (y > 1):
            y = 1

        return x, y
