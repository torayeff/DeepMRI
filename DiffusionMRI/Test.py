import sys
sys.path.append('/home/agajan/DeepMRI')
import torch
import torch.nn as nn
from deepmri import Datasets, utils
from DiffusionMRI.ConvEncoder import ConvEncoder
from DiffusionMRI.ConvTransposeDecoder import ConvTransposeDecoder
import time
import nibabel as nib
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # set False whenever input size varies
print("Device: ", device)

batch_size = 1
data_path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/agajan/experiment_DiffusionMRI/data/'
# trainset = Datasets.MRIDataset(data_path, normalize=False)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

encoder = ConvEncoder(input_channels=288)
encoder.to(device)
encoder.train()

decoder = ConvTransposeDecoder(out_channels=288)
decoder.to(device)
decoder.train()
# utils.count_model_parameters(encoder)

dmri_path = os.path.join(data_path, 'data_168240.nii.gz')
print("Loading dmri")
x = nib.load(dmri_path).get_fdata()
input("Press to transpose")
x = x.transpose(3, 0, 1, 2)
input("Press to convert to tensor")
x = torch.tensor(x).float().unsqueeze(0)
print(x.shape)
input("Press continue to load to GPU")
x = x.to(device)
input("Press to encode")
feature = encoder(x)
print("Feature shape: ", feature.shape)
input("Press to decode")
out = decoder(feature)
print(out.shape)
input("Press to continue")
