import sys
import time
import torch
import torch.nn as nn

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, utils  # noqa: E402
from DiffusionMRI.ConvEncoder import ConvEncoder  # noqa: E402
from DiffusionMRI.ConvDecoder import ConvDecoder  # noqa: E402

script_start = time.time()

# gpu settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # set False whenever input size varies
print("Device: ", device)

# deterministic
# torch.manual_seed(0)
# torch.backends.cudnn.benchmark = False  # set False whenever input size varies
# torch.backends.cudnn.deterministic = True

# training data settings
mu = 307.3646240234375
std = 763.4876098632812
batch_size = 16

data_path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/agajan/experiment_DiffusionMRI/'
trainset = Datasets.Volume3dDataset(data_path + 'tensors_3d/', mu=mu, std=std, normalize=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
print("Total training examples: ", len(trainset))

# model settings
encoder = ConvEncoder(input_channels=1)
encoder.to(device)

decoder = ConvDecoder(out_channels=1)
decoder.to(device)


# load pretrained weights
prev_epoch = 23
encoder_path = '/home/agajan/DeepMRI/DiffusionMRI/models/conv_encoder_epoch_' + str(prev_epoch)
decoder_path = '/home/agajan/DeepMRI/DiffusionMRI/models/conv_decoder_epoch_' + str(prev_epoch)
encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))

p1 = utils.count_model_parameters(encoder)
p2 = utils.count_model_parameters(decoder)
print("Total parameters: {}, trainable parameters: {}".format(p1[0] + p2[0], p1[1] + p2[1]))

# criterion and optimizer settings
criterion = nn.MSELoss()
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.5)

# training
encoder.train()
decoder.train()

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

        running_loss = running_loss + loss.item() * data.size(0)
        # if iters % 1 == 0:
        #     print("Iter #{}, iter time: {:.5f}, batch loss: {}".format(iters, time.time() - iter_time, loss.item()))
        iters += 1

    if epoch % 1 == 0:
        torch.save(encoder.state_dict(), "models/conv_encoder_epoch_{}".format(epoch + prev_epoch))
        torch.save(decoder.state_dict(), "models/conv_decoder_epoch_{}".format(epoch + prev_epoch))

    epoch_loss = running_loss / len(trainset)

    # scheduler.step(epoch_loss)
    print("Epoch #{}/{},  epoch loss: {}, epoch time: {:.5f} seconds".format(epoch + prev_epoch, num_epochs, epoch_loss,
                                                                             time.time() - epoch_start))

print("Total running time: {}".format(time.time() - script_start))
