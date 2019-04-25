import torch
import torch.nn as nn
import Datasets
from RNNEncoder import RNNEncoder
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

torch.backends.cudnn.benchmark = True  # set False whenever input size varies

batch_size = 67
trainset = Datasets.ADHDFeatureDataset('/home/agajan/feature_tensors_4d/train/',
                                       csv_file='/home/agajan/DeepMRI/adhd_trainset.csv',
                                       seq_len=50,
                                       binary=True)
validset = Datasets.ADHDFeatureDataset('/home/agajan/feature_tensors_4d/valid/',
                                       csv_file='/home/agajan/DeepMRI/adhd_trainset.csv',
                                       seq_len=50,
                                       binary=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=6)

print("Total training examples: ", len(trainset))
print("Total validation examples: ", len(validset))

for data in validloader:
    break
