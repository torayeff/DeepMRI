import sys
import torch
import numpy as np
from os.path import join

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets  # noqa: E402
from DiffusionMRI.models_bkp2.Model5 import Encoder, Decoder  # noqa: E402

# ------------------------------------------Settings--------------------------------------------------------------------
experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
subj_id = '784565'
data_path = join(experiment_dir, 'tractseg_data', subj_id)
model_name = 'Model5'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
deterministic = True  # reproducibility
seed = 0  # random seed for reproducibility
if deterministic:
    torch.manual_seed(seed)
torch.backends.cudnn.benchmark = (not deterministic)  # set False whenever input size varies
torch.backends.cudnn.deterministic = deterministic

batch_size = 2 ** 15
start_epoch = 100
channels = 50
noise_prob = None
trainset = Datasets.VoxelDataset(data_path,
                                 file_name='data.nii.gz',
                                 normalize=False,
                                 scale=True)
total_examples = len(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=6)
print("Total training examples: {}, Batch size: {}, Iters per epoch: {}".format(total_examples,
                                                                                batch_size,
                                                                                len(trainloader)))

# ------------------------------------------Model-----------------------------------------------------------------------
# model settings
encoder = Encoder()
decoder = Decoder()
encoder.to(device)
decoder.to(device)
encoder.eval()
decoder.eval()

encoder_path = "{}/saved_models/{}_encoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
decoder_path = "{}/saved_models/{}_decoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))
print("Loaded pretrained weights starting from epoch {}".format(start_epoch))

learned_features = np.zeros((145, 174, 145, channels))
c = 0
with torch.no_grad():
    for data in trainloader:
        f = encoder(data['data'].to(device))
        for b in range(f.shape[0]):
            crd_0 = data['coord'][0][b].item()
            crd_1 = data['coord'][1][b].item()
            crd_2 = data['coord'][2][b].item()
            fvec = f[b].detach().cpu().squeeze().numpy().reshape(channels)
            learned_features[crd_0, crd_1, crd_2] = fvec
        c += 1
        print(c, end=" ")

save_path = join(data_path, 'learned_features/{}_features_epoch_{}.npz'.format(model_name, start_epoch))
np.savez(save_path, data=learned_features)
