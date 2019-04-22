import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
import pickle
import utils


class FMRIDataset(Dataset):
    """Functional MRI dataset."""

    def __init__(self, root_dir):
        """
        Args:
            root_dir: Directory with all fMRI images.
        """

        self.file_paths = []
        for file_name in os.listdir(root_dir):
            if file_name.endswith('.nii.gz'):
                self.file_paths.append(os.path.join(root_dir, file_name))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Fetches fMRI images."""

        x = nib.load(self.file_paths[idx]).get_fdata().transpose((3, 0, 1, 2))  # time, x, y, z
        x = torch.tensor(x).unsqueeze(1).float()  # add channel dimension, 5D data = time, channel, x, y, z

        return x


class Slice3dDataset(Dataset):
    """3D slice dataset"""

    def __init__(self, root_dir):
        """
        Args:
            root_dir: Directory with all 3d slices.
        """

        self.file_paths = []
        for file_name in os.listdir(root_dir):
            if file_name.endswith('.nii.gz'):
                self.file_paths.append(os.path.join(root_dir, file_name))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Fetches 3d slices."""

        with open(self.file_paths[idx], "rb") as f:
            x = pickle.load(f)

        return x


trainset = Slice3dDataset('/home/agajan/3d_np_data/')
print(len(trainset))

print(trainset[0].shape)
