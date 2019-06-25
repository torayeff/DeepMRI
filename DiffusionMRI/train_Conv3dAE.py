import sys
import time
import torch
import numpy as np

sys.path.append('/home/agajan/DeepMRI')
from deepmri import utils  # noqa: E402
from DiffusionMRI.Conv3dAE import ConvEncoder, ConvDecoder  # noqa: E402

script_start = time.time()

# ------------------------------------------Settings--------------------------------------------------------------------
experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
data_path = experiment_dir + 'tractseg_data/784565/shore/shore_coefficients_radial_border_2.npz'
model_name = "Conv3dAE"
# ------------------------------------------Model-----------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
deterministic = False  # reproducibility
seed = 0  # random seed for reproducibility
if deterministic:
    torch.manual_seed(seed)
torch.backends.cudnn.benchmark = (not deterministic)  # set False whenever input size varies
torch.backends.cudnn.deterministic = deterministic

start_epoch = 0  # for loading pretrained weights
num_epochs = 10000  # number of epochs to trains
checkpoint = 1000  # save model every checkpoint epoch
# ------------------------------------------Data------------------------------------------------------------------------
data = np.load(data_path)['data']
# mu = data.mean()
# std = data.std()
# data = (data - mu) / std
data = data.transpose(3, 0, 1, 2)
data = torch.tensor(data).float().unsqueeze(0)
print(data.shape)
trainloader = [{'data': data}]
# ------------------------------------------Model-----------------------------------------------------------------------
# model settings
encoder = ConvEncoder()
decoder = ConvDecoder()
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
criterion = torch.nn.MSELoss()
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.003)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       verbose=True,
                                                       min_lr=1e-6,
                                                       patience=5)
# ------------------------------------------Training--------------------------------------------------------------------
print("Training: {}".format(model_name))
# utils.evaluate_ae(encoder, decoder, criterion, device, trainloader, masked_loss=True)
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
               scheduler=None,
               checkpoint=checkpoint,
               print_iter=False,
               eval_epoch=1000,
               masked_loss=False)

print("Total running time: {}".format(time.time() - script_start))
