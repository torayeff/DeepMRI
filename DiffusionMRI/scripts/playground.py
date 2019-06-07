import sys
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, CustomLosses, utils  # noqa: E402
from DiffusionMRI.Conv2dAESagittal import ConvEncoder, ConvDecoder  # noqa: E402

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
data_path = experiment_dir + 'tractseg_data/train/sagittal/'
model_name = "SagittalConv2dAE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
batch_size = 16

trainset = Datasets.OrientationDatasetChannelNorm(data_path, normalize=True, bg_zero=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=10)
print(len(trainset))

encoder = ConvEncoder(input_channels=288)
decoder = ConvDecoder(out_channels=288)
encoder.to(device)
decoder.to(device)
start_epoch = 2200
# start_epoch = 0
if start_epoch != 0:
    encoder_path = "{}/models/{}_encoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
    decoder_path = "{}/models/{}_decoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    print("Loaded pretrained weights starting from epoch {}".format(start_epoch))

criterion = CustomLosses.MaskedMSE()
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.0003)

num_epochs = 1
print_iter = False
utils.evaluate_ae(encoder, decoder, criterion, device, trainloader, masked_loss=True)
for epoch in range(1, num_epochs + 1):
    encoder.train()
    decoder.train()
    epoch_start = time.time()
    total_examples = 0
    running_loss = 0.0

    iters = 1

    for batch in trainloader:
        iter_time = time.time()

        # zero gradients
        optimizer.zero_grad()

        # forward
        x = batch['data'].to(device)
        out = decoder(encoder(x))

        # calculate loss
        mask = batch['mask'].unsqueeze(1).to(device)
        loss = criterion(x, out, mask)

        # backward
        loss.backward()

        # update params
        optimizer.step()

        # track loss
        running_loss = running_loss + loss.item() * batch['data'].size(0)
        total_examples += batch['data'].size(0)
        if print_iter:
            print("Iteration #{}, loss: {}, iter time: {}".format(iters, loss.item(), time.time() - iter_time))

        named_parameters = list(encoder.named_parameters()) + list(decoder.named_parameters())
        utils.plot_grad_flow(named_parameters)

        iters += 1

    epoch_loss = running_loss / total_examples
    print("Epoch #{}/{},  epoch loss: {}, epoch time: {:.5f} seconds".format(epoch + start_epoch,
                                                                             num_epochs,
                                                                             epoch_loss,
                                                                             time.time() - epoch_start))
plt.show()
