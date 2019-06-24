import sys
import torch
import numpy as np

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, utils  # noqa: E402
from DiffusionMRI.NHConv3dAE import ConvEncoder, ConvDecoder  # noqa: E402

# ------------------------------------------Settings--------------------------------------------------------------------
experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
data_path = experiment_dir + 'tractseg_data/784565/'
model_name = 'NHConv3dAE3x3x3'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
deterministic = False  # reproducibility
seed = 0  # random seed for reproducibility
if deterministic:
    torch.manual_seed(seed)
torch.backends.cudnn.benchmark = (not deterministic)  # set False whenever input size varies
torch.backends.cudnn.deterministic = deterministic

trainset = Datasets.NeighborhoodDataset(data_path, normalize=False)
total_examples = len(trainset)

batch_size = 2**15
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=6)
print("Total training examples: {}, Batch size: {}, Iters per epoch: {}".format(total_examples,
                                                                                batch_size,
                                                                                total_examples / batch_size))

# ------------------------------------------Model-----------------------------------------------------------------------
# model settings
encoder = ConvEncoder()
decoder = ConvDecoder()
encoder.to(device)
decoder.to(device)

start_epoch = 20
encoder_path = "{}/models/{}_encoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
decoder_path = "{}/models/{}_decoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))
print("Loaded pretrained weights starting from epoch {}".format(start_epoch))

nh_features = np.zeros((145, 174, 145, 288))
c = 0
with torch.no_grad():
    for data in trainloader:
        f = encoder(data['data'].to(device))
        for b in range(f.shape[0]):
            crd_0 = data['coord'][0][b].item()
            crd_1 = data['coord'][1][b].item()
            crd_2 = data['coord'][2][b].item()
            fvec = f[b].detach().cpu().squeeze().numpy().reshape(288)
            nh_features[crd_0, crd_1, crd_2] = fvec
        c += 1
        print(c, end=" ")

save_path = data_path + 'learned_features/nh_features_epoch_{}.npz'.format(start_epoch)
np.savez(save_path, data=nh_features)
