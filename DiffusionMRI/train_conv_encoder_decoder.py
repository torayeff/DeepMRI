import sys
import time
import torch
import torch.nn as nn

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, utils  # noqa: E402
from DiffusionMRI.ConvEncoder import ConvEncoder  # noqa: E402
from DiffusionMRI.ConvTransposeDecoder import ConvTransposeDecoder  # noqa: E402

script_start = time.time()

# gpu settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # set False whenever input size varies
print("Device: ", device)

# training data settings
mu = 307.3646240234375
std = 763.4876098632812
batch_size = 16
data_path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/agajan/experiment_DiffusionMRI/'
trainset = Datasets.Volume3dDataset(data_path + 'tensors_3d/', mu=mu, std=std)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
print("Total training examples: ", len(trainset))

# model settings
encoder = ConvEncoder(input_channels=1)
encoder.to(device)
encoder.train()

decoder = ConvTransposeDecoder(out_channels=1)
decoder.to(device)
decoder.train()

# load pretrained weights
prev_epoch = 12
encoder_path = '/home/agajan/DeepMRI/DiffusionMRI/models/adj_transpose_conv_encoder_epoch_' + str(prev_epoch)
decoder_path = '/home/agajan/DeepMRI/DiffusionMRI/models/adj_transpose_conv_decoder_epoch_' + str(prev_epoch)
encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))

p1 = utils.count_model_parameters(encoder)
p2 = utils.count_model_parameters(decoder)
print("Total parameters: {}, trainable parameters: {}".format(p1[0] + p2[0], p1[1] + p2[1]))

# criterion and optimizer settings
criterion = nn.MSELoss()
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.5)

# training
num_epochs = 100
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
        if iters % 100 == 0:
            print("Iter #{}, iter time: {:.5f}, batch loss: {}".format(iters, time.time() - iter_time, loss.item()))

            torch.save(encoder.state_dict(), "models/adj_transpose_conv_encoder_epoch_{}_iter_{}"
                       .format(epoch + prev_epoch, iters))
            torch.save(decoder.state_dict(), "models/adj_transpose_conv_decoder_epoch_{}_iter_{}"
                       .format(epoch + prev_epoch, iters))
        iters += 1

    if epoch % 1 == 0:
        torch.save(encoder.state_dict(), "models/adj_transpose_conv_encoder_epoch_{}".format(epoch + prev_epoch))
        torch.save(decoder.state_dict(), "models/adj_transpose_conv_decoder_epoch_{}".format(epoch + prev_epoch))

    epoch_loss = running_loss / len(trainset)

    scheduler.step(epoch_loss)
    print("Epoch #{}/{},  epoch loss: {}, epoch time: {:.5f} seconds".format(epoch + prev_epoch, num_epochs, epoch_loss,
                                                                             time.time() - epoch_start))

print("Total running time: {}".format(time.time() - script_start))
