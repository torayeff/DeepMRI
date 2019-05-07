import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import pickle
import pandas as pd


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
            x = x.unsqueeze(0)  # 3D Volume: channel x width x height x depth

        if self.normalize:
            if self.mu is None:
                x = (x - x.mean()) / x.std()
            else:
                x = (x - self.mu) / self.std

        return x


class Slice3dDataset(Dataset):
    """3D slice dataset"""

    def __init__(self, root_dir, normalize=True):
        """
        Args:
            root_dir: Directory with all 3d slices.
        """
        self.file_paths = []
        self.normalize = normalize
        for file_name in os.listdir(root_dir):
            if file_name.endswith('.3dtensor'):
                self.file_paths.append(os.path.join(root_dir, file_name))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Fetches 3d slices."""

        with open(self.file_paths[idx], "rb") as f:
            x = pickle.load(f)  # channel=1 x w x h x

        if self.normalize:
            x = (x - x.mean()) / x.std()
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
