import sys
import time
import torch
import numpy as np

sys.path.append('/home/agajan/DeepMRI')
from DiffusionMRI.Conv3dAE import ConvEncoder, ConvDecoder  # noqa: E402

script_start = time.time()

# ------------------------------------------Settings--------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
subj_id = '784565'
epoch = 1000
data_path = experiment_dir + 'tractseg_data/{}/shore_coefficients_radial_border_2.npz'.format(subj_id)
save_path = experiment_dir + 'tractseg_data/{}/learned_features/shore_features_epoch_{}.npz'.format(subj_id, epoch)
model_name = "Conv3dAE"
# ------------------------------------------Model-----------------------------------------------------------------------
encoder = ConvEncoder()
decoder = ConvDecoder()
encoder.to(device)
decoder.to(device)

encoder_path = '{}/models/{}_encoder_epoch_{}'.format(experiment_dir, model_name, epoch)
decoder_path = '{}/models/{}_decoder_epoch_{}'.format(experiment_dir, model_name, epoch)
encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))
print('Loaded pretrained weights rom epoch {}'.format(epoch))
# ------------------------------------------Data------------------------------------------------------------------------
data = np.load(data_path)['data']
data = data.transpose(3, 0, 1, 2)
data = torch.tensor(data).float().unsqueeze(0)
# ------------------------------------------Create Features-------------------------------------------------------------
criterion = torch.nn.MSELoss()
with torch.no_grad():
    encoder.eval()
    decoder.eval()
    x = data.to(device)
    features = encoder(x)
    y = decoder(features)
    print('Reconstruction loss: {}'.format(criterion(x, y).item()))
    # print('Loss between x and f: {}'.format(criterion(x, features).item()))
    features = features.detach().cpu().squeeze().numpy()
    features = features.transpose(1, 2, 3, 0)
    print(features.shape)
    np.savez(save_path, data=features)
