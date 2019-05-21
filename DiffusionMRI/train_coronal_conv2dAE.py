import sys
import time
import torch
import torch.nn as nn

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, utils  # noqa: E402
from DiffusionMRI.CoronalConv2dAE import ConvEncoder  # noqa: E402
from DiffusionMRI.CoronalConv2dAE import ConvDecoder  # noqa: E402

script_start = time.time()

# ------------------------------------------Settings--------------------------------------------------------------------
experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
data_path = experiment_dir + 'data/train/coronal_part1/'
model_name = "CoronalConv2dAE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
deterministic = False  # reproducibility
seed = 0  # random seed for reproducibility
if deterministic:
    torch.manual_seed(seed)
torch.backends.cudnn.benchmark = (not deterministic)  # set False whenever input size varies
torch.backends.cudnn.deterministic = deterministic

# data
mu = 368.62549
std = 823.93335
batch_size = 16

start_epoch = 0  # for loading pretrained weights
num_epochs = 100  # number of epochs to trains
checkpoint = 1  # save model every checkpoint epoch
# ------------------------------------------Data------------------------------------------------------------------------
trainset = Datasets.OrientationDataset(data_path, mu=mu, std=std, normalize=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
total_examples = len(trainset)
print("Total training examples: {}, Batch size: {}, Iters per epoch: {}".format(total_examples,
                                                                                batch_size,
                                                                                total_examples // batch_size + 1))
# ------------------------------------------Model-----------------------------------------------------------------------
# model settings
encoder = ConvEncoder(input_channels=288)
decoder = ConvDecoder(out_channels=288)
encoder.to(device)
decoder.to(device)

if start_epoch != 0:
    encoder_path = "{}/models/{}_encoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
    decoder_path = "{}/models/{}_decoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    print("Loaded pretrained weights starting from epoch {}".format(start_epoch))

p1 = utils.count_model_parameters(encoder)
p2 = utils.count_model_parameters(decoder)
print("Total parameters: {}, trainable parameters: {}".format(p1[0] + p2[0], p1[1] + p2[1]))

# criterion and optimizer settings
criterion = nn.MSELoss()
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters, lr=3e-4)
# ------------------------------------------Training--------------------------------------------------------------------


utils.train_ae(encoder,
               decoder,
               criterion,
               optimizer,
               device,
               trainloader,
               num_epochs,
               model_name,
               experiment_dir,
               start_epoch=start_epoch,
               checkpoint=checkpoint,
               print_iter=False)

print("Total running time: {}".format(time.time() - script_start))
