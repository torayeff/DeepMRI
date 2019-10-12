import sys
import time
import torch

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, utils  # noqa: E402
from DiffusionMRI.models.VAE import Encoder, Decoder  # noqa: E402

script_start = time.time()

# ------------------------------------------Settings--------------------------------------------------------------------
experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
data_path = experiment_dir + 'tractseg_data/784565/'
model_name = "VAE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
deterministic = True  # reproducibility
seed = 0  # random seed for reproducibility
if deterministic:
    torch.manual_seed(seed)
torch.backends.cudnn.benchmark = (not deterministic)  # set False whenever input size varies
torch.backends.cudnn.deterministic = deterministic

# data
batch_size = 2 ** 8

start_epoch = 0  # for loading pretrained weights
num_epochs = 10  # number of epochs to trains
checkpoint = 10  # save model every checkpoint epoch
# ------------------------------------------Data------------------------------------------------------------------------

trainset = Datasets.VoxelDataset(data_path,
                                 file_name='data.nii.gz',
                                 normalize=False,
                                 scale=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
total_examples = len(trainset)
print("Total training examples: {}, Batch size: {}, Iters per epoch: {}".format(total_examples,
                                                                                batch_size,
                                                                                total_examples / batch_size))
# ------------------------------------------Model-----------------------------------------------------------------------
# model settings
encoder = Encoder(288, 10, 10, device)
decoder = Decoder(288, 10, 10)
encoder.to(device)
decoder.to(device)

if start_epoch != 0:
    encoder_path = "{}saved_models/{}_encoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
    decoder_path = "{}saved_models/{}_decoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    print("Loaded pretrained weights starting from epoch {}".format(start_epoch))

p1 = utils.count_model_parameters(encoder)
p2 = utils.count_model_parameters(decoder)
print("Total parameters: {}, trainable parameters: {}".format(p1[0] + p2[0], p1[1] + p2[1]))

criterion = torch.nn.MSELoss()
# criterion = torch.nn.BCEWithLogitsLoss()

parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters)
# ------------------------------------------Training--------------------------------------------------------------------
start_time = time.time()
for epoch in range(1, num_epochs+1):
    for batch_idx, batch in enumerate(trainloader):

        # don't need labels, only the images (features)
        features = batch['data'].to(device)

        z_mean, z_log_var, encoded = encoder(features)
        decoded = decoder(encoded)

        # cost = reconstruction loss + Kullback-Leibler divergence
        kl_divergence = (0.5 * (z_mean ** 2 +
                                torch.exp(z_log_var) - z_log_var - 1)).sum()
        pixelwise_bce = torch.nn.BCEWithLogitsLoss(reduction='sum')(decoded, features)
        cost = kl_divergence + pixelwise_bce

        optimizer.zero_grad()
        cost.backward()

        optimizer.step()

        if not batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                  % (epoch + 1, num_epochs, batch_idx,
                     len(trainloader), cost))

    if epoch % checkpoint == 0:
        torch.save(encoder.state_dict(), "{}saved_models/{}_encoder_epoch_{}".format(experiment_dir,
                                                                                     model_name,
                                                                                     epoch + start_epoch))
        torch.save(decoder.state_dict(), "{}saved_models/{}_decoder_epoch_{}".format(experiment_dir,
                                                                                     model_name,
                                                                                     epoch + start_epoch))

    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
