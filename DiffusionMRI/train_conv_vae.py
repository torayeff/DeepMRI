import sys
import time
import torch
import torch.nn.functional as F

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, CustomLosses, utils  # noqa: E402
from DiffusionMRI.ConvVAE import ConvVAE  # noqa: E402

script_start = time.time()

# ------------------------------------------Settings--------------------------------------------------------------------
experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
data_path = experiment_dir + 'tractseg_data/784565/training_slices/coronal/'
model_name = "Model15"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
deterministic = False  # reproducibility
seed = 0  # random seed for reproducibility
if deterministic:
    torch.manual_seed(seed)
torch.backends.cudnn.benchmark = (not deterministic)  # set False whenever input size varies
torch.backends.cudnn.deterministic = deterministic

# data
batch_size = 8

# noise probability
noise_prob = None

start_epoch = 0  # for loading pretrained weights
num_epochs = 200  # number of epochs to trains
checkpoint = 200  # save model every checkpoint epoch
# ------------------------------------------Data------------------------------------------------------------------------

trainset = Datasets.OrientationDataset(data_path,
                                       scale=True,
                                       normalize=False,
                                       bg_zero=True,
                                       noise_prob=noise_prob,
                                       alpha=1)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
total_examples = len(trainset)
print("Total training examples: {}, Batch size: {}, Iters per epoch: {}".format(total_examples,
                                                                                batch_size,
                                                                                total_examples / batch_size))
# ------------------------------------------Model-----------------------------------------------------------------------
# model settings
model = ConvVAE(128, device)
model.to(device)

if start_epoch != 0:
    model_path = "{}/saved_models/{}_epoch_{}".format(experiment_dir, model_name, start_epoch)
    model.load_state_dict(torch.load(model_path))
    print("Loaded pretrained weights starting from epoch {}".format(start_epoch))

p1 = utils.count_model_parameters(model)
print("Total parameters: {}, trainable parameters: {}".format(p1[0], p1[1]))

optimizer = torch.optim.Adam(model.parameters())
# ------------------------------------------Training--------------------------------------------------------------------
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    for batch_idx, batch in enumerate(trainloader):

        # don't need labels, only the images (features)
        features = batch['data'].to(device)

        z_mean, z_log_var, encoded, decoded = model(features)

        # cost = reconstruction loss + Kullback-Leibler divergence
        kl_divergence = (0.5 * (z_mean ** 2 +
                                torch.exp(z_log_var) - z_log_var - 1)).sum()
        pixelwise_bce = F.binary_cross_entropy(decoded, features, reduction='sum')
        cost = kl_divergence + pixelwise_bce

        optimizer.zero_grad()
        cost.backward()

        optimizer.step()

        if not batch_idx % 50:
            print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                  % (epoch + 1, num_epochs, batch_idx,
                     len(trainloader), cost))
    if epoch % checkpoint == 0:
        torch.save(model.state_dict(), "{}saved_models/{}_epoch_{}".format(experiment_dir,
                                                                           model_name,
                                                                           epoch + start_epoch))

    print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))