import sys
import torch
import numpy as np
import os

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets  # noqa: E402
# from DiffusionMRI.Conv2dAEFullSpatial import ConvEncoder  # noqa: E402
from DiffusionMRI.Conv2dAESagittal import ConvEncoder  # noqa: E402

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
torch.backends.cudnn.benchmark = True  # set False whenever input size varies

subj_id = '784565'
# subj_id = '786569'
experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
# orients = ['sagittal', 'coronal', 'axial']
# model_names = ["SagittalConv2dAEFullSpatial", "CoronalConv2dAEFullSpatial", "AxialConv2dAEFullSpatial"]
orients = ['sagittal']
model_names = ["SagittalConv2dAE"]
feature_shapes = [(145, 22, 19, 128),
                  (174, 19, 19, 128),
                  (145, 19, 22, 128)]
encoder = ConvEncoder(input_channels=288)
encoder.to(device)
encoder.eval()

for i, orient in enumerate(orients):
    print("Processing {} features".format(orient))
    # data_path = os.path.join(experiment_dir, 'tractseg_data', subj_id, 'orients', orient)
    data_path = experiment_dir + 'tractseg_data/train/sagittal/'
    features_save_path = os.path.join(experiment_dir, 'tractseg_data', subj_id, 'orient_features')

    # mu = 453.9321075958082
    # std = 969.7367041395971
    # dataset = Datasets.OrientationDataset(data_path, mu=mu, std=std, normalize=True, sort_fns=True)
    dataset = Datasets.OrientationDatasetChannelNorm(data_path, normalize=True, sort_fns=True, bg_zero=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)

    epoch = 2200
    encoder_path = "{}/models/{}_encoder_epoch_{}".format(experiment_dir, model_names[i], epoch)
    encoder.load_state_dict(torch.load(encoder_path))
    print("Loaded pretrained weights starting from epoch {} for {}".format(epoch, model_names[i]))

    with torch.no_grad():
        orient_features = torch.zeros(feature_shapes[i])
        for j, data in enumerate(dataloader):
            x = data['data'].to(device)
            feature = encoder(x).detach().cpu().squeeze().permute(1, 2, 0)
            idx = int(data['file_name'][0][:-4][-3:])
            orient_features[idx] = feature
            print(idx)

        orient_features = orient_features.numpy()
        np.savez(os.path.join(features_save_path, 'new_strided_{}_features.npz'.format(orient)), data=orient_features)
