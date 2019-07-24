import sys
import torch
import numpy as np
import os

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets  # noqa: E402
from DiffusionMRI.ConvVAE import ConvVAE  # noqa: E402

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
torch.backends.cudnn.benchmark = True  # set False whenever input size varies

experiment_dir = '/home/agajan/experiment_DiffusionMRI/'

subj_id = '784565'
orients = ['coronal']
model_name = "Model15"
feature_shapes = [(174, 145, 145, 22)]
epoch = 200
noise_prob = None
num_latent = 128

model = ConvVAE(num_latent, device)
model.to(device)
model.eval()

for i, orient in enumerate(orients):
    print("Processing {} features".format(orient))
    data_path = os.path.join(experiment_dir, 'tractseg_data', subj_id, 'training_slices', orient)
    features_save_path = os.path.join(experiment_dir, 'tractseg_data', subj_id, 'learned_features')

    dataset = Datasets.OrientationDataset(data_path, scale=True, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)

    model_path = "{}saved_models/{}_epoch_{}".format(experiment_dir, model_name, epoch)
    model.load_state_dict(torch.load(model_path))
    print("Loaded pretrained weights starting from epoch {} for {}".format(epoch, model_name))

    with torch.no_grad():

        orient_features = torch.zeros(feature_shapes[i])

        for j, data in enumerate(dataloader):

            feature = data['data'].to(device)
            z_mean, z_log_var, encoded = model.encoder(feature)
            feature = model.decoder(encoded, return_feature=True)

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
