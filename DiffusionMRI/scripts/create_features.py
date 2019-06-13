import sys
import torch
import numpy as np
import os

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets  # noqa: E402
from DiffusionMRI.Conv2dAEFullSpatial import ConvEncoder  # noqa: E402

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
torch.backends.cudnn.benchmark = True  # set False whenever input size varies

subj_id = '786569'
experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
orients = ['coronal']
model_names = ["Conv2dAEFullSpatial"]
feature_shapes = [(174, 145, 145, 36)]
encoder = ConvEncoder()
encoder.to(device)
encoder.eval()

for i, orient in enumerate(orients):
    print("Processing {} features".format(orient))
    data_path = os.path.join(experiment_dir, 'tractseg_data', subj_id, 'orientation_slices', orient)
    features_save_path = os.path.join(experiment_dir, 'tractseg_data', subj_id, 'learned_features')

    dataset = Datasets.OrientationDatasetChannelNorm(data_path, normalize=True, sort_fns=True, bg_zero=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)

    epoch = 2
    encoder_path = "{}models/{}_encoder_epoch_{}".format(experiment_dir, model_names[i], epoch)
    encoder.load_state_dict(torch.load(encoder_path))
    print("Loaded pretrained weights starting from epoch {} for {}".format(epoch, model_names[i]))

    with torch.no_grad():
        orient_features = torch.zeros(feature_shapes[i])

        for j, data in enumerate(dataloader):
            x = data['data'].to(device)
            feature = encoder(x)
            orient_feature = feature.detach().cpu().squeeze().permute(1, 2, 0)

            idx = int(data['file_name'][0][:-4][-3:])
            orient_features[idx] = orient_feature
            print(idx)

        orient_features = orient_features.numpy()
        np.savez(os.path.join(features_save_path, 'full_{}_features_epoch_{}.npz'.format(orient, epoch)),
                 data=orient_features)
