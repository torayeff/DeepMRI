import sys
import time
import torch
import torch.nn as nn

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, utils  # noqa: E402
from DiffusionMRI.ConvEncoder import ConvEncoder  # noqa: E402
from DiffusionMRI.ConvUpsampleDecoder import ConvUpsampleDecoder  # noqa: E402
from DiffusionMRI.ConvTransposeDecoder import ConvTransposeDecoder  # noqa: E402


# gpu settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # set False whenever input size varies
print("Device: ", device)

# training data settings
mu = 307.3646240234375
std = 763.4876098632812
batch_size = 8
data_path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/agajan/experiment_DiffusionMRI/'
trainset = Datasets.Volume3dDataset(data_path + 'tensors_3d/', mu=mu, std=std)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)
print("Total training examples: ", len(trainset))

# model settings
encoder = ConvEncoder(input_channels=1)
encoder.to(device)
encoder.train()

decoder = ConvUpsampleDecoder(out_channels=1)
# decoder = ConvTransposeDecoder(out_channels=1)
decoder.to(device)
decoder.train()

p1 = utils.count_model_parameters(encoder)
p2 = utils.count_model_parameters(decoder)
print("Total parameters: {}, trainable parameters: {}".format(p1[0] + p2[0], p1[1] + p2[1]))

# criterion and optimizer settings
criterion = nn.MSELoss()
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters)

# training
num_epochs = 500
iters = 1
print("Training started for {} epochs".format(num_epochs))
for epoch in range(1, num_epochs + 1):
    running_loss = 0.0
    epoch_start = time.time()

    for data in trainloader:
        iter_time = time.time()

        optimizer.zero_grad()
        x = data.to(device)

        out = decoder(encoder(x))

        loss = criterion(x, out)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        if iters % 1000 == 0:
            print("Iter #{}, iter time: {:.5f}, batch loss: {}".format(iters, time.time() - iter_time, loss.item()))
        iters += 1

    if epoch % 1 == 0:
        torch.save(encoder.state_dict(), "models/upsample_conv_encoder_epoch_{}".format(epoch))
        torch.save(decoder.state_dict(), "models/upsample_conv_decoder_epoch_{}".format(epoch))

    epoch_loss = running_loss / len(trainset)
    print("Epoch #{}/{},  epoch loss: {}, epoch time: {:.5f} seconds".format(epoch, num_epochs, epoch_loss,
                                                                             time.time() - epoch_start))
