import sys
import torch
import numpy as np
import os

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets  # noqa: E402
from DiffusionMRI.Conv2dAEStrided import ConvEncoder  # noqa: E402

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
torch.backends.cudnn.benchmark = True  # set False whenever input size varies

experiment_dir = '/home/agajan/experiment_DiffusionMRI/'

subj_id = '784565'
orients = ['coronal']
model_name = "Conv2dAECoronalStrided_SHORE"
feature_shapes = [(174, 145, 145, 14)]
epoch = 200

encoder = ConvEncoder(input_size=(145, 145))
encoder.to(device)
encoder.eval()

for i, orient in enumerate(orients):
    print("Processing {} features".format(orient))
    data_path = os.path.join(experiment_dir, 'tractseg_data', subj_id, 'shore_slices', orient)
    features_save_path = os.path.join(experiment_dir, 'tractseg_data', subj_id, 'learned_features')

    dataset = Datasets.OrientationDatasetChannelNorm(data_path, normalize=True, sort_fns=True, bg_zero=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)

    encoder_path = "{}models/{}_encoder_epoch_{}".format(experiment_dir, model_name, epoch)
    encoder.load_state_dict(torch.load(encoder_path))
    print("Loaded pretrained weights starting from epoch {} for {}".format(epoch, model_name))

    with torch.no_grad():
        orient_features = torch.zeros(feature_shapes[i])

        for j, data in enumerate(dataloader):
            x = data['data'].to(device)
            feature = encoder(x, return_all=True)
            # print(feature[0].shape, feature[1].shape, feature[2].shape)
            feature = feature[1]
            # feature = torch.cat((feature[0], feature[2]), dim=1)
            print(feature.shape)
            orient_feature = feature.detach().cpu().squeeze().permute(1, 2, 0)

            idx = int(data['file_name'][0][:-4][-3:])
            orient_features[idx] = orient_feature
            print(idx)

        orient_features = orient_features.numpy()
        # transpose features
        if orient == 'coronal':
            orient_features = orient_features.transpose(1, 0, 2, 3)
        if orient == 'axial':
            orient_features = orient_features.transpose(1, 2, 0, 3)

        np.savez(os.path.join(features_save_path, '{}_features_epoch_{}.npz'.format(model_name, epoch)),
                 data=orient_features)
