import sys
import torch
import numpy as np
import os

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, utils  # noqa: E402
from DiffusionMRI.Conv2dAESagittalFullSpatial import ConvEncoder  # noqa: E402

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
torch.backends.cudnn.benchmark = True  # set False whenever input size varies

subj_id = '786569'
experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
data_path = os.path.join(experiment_dir, 'tractseg_data', subj_id, 'orients/sagittal')
features_save_path = os.path.join(experiment_dir, 'tractseg_data', subj_id, 'orient_features')
model_name = "SagittalConv2dAEFullSpatial"

mu = 453.9321075958082
std = 969.7367041395971
dataset = Datasets.OrientationDataset(data_path, mu=mu, std=std, normalize=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)

encoder = ConvEncoder(input_channels=288)
encoder.to(device)

epoch = 200
encoder_path = "{}/models/{}_encoder_epoch_{}".format(experiment_dir, model_name, epoch)
encoder.load_state_dict(torch.load(encoder_path))
print("Loaded pretrained weights starting from epoch {}".format(epoch))

orient_features = torch.zeros(145, 174, 145, 36)

with torch.no_grad():
    for data in dataloader:
        x = data['data'].to(device)
        feature = encoder(x).detach().cpu().squeeze().permute(1, 2, 0)
        idx = int(data['file_name'][0][:-4][-3:])
        orient_features[idx] = feature
        print(idx)

    orient_features = orient_features.numpy()
    np.savez(os.path.join(features_save_path, 'sagittal_features_145x174x145x36.npz'), data=orient_features)
