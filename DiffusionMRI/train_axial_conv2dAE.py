import sys
import time
import torch
import torch.nn as nn

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, utils  # noqa: E402
from DiffusionMRI.AxialConv2dAE import ConvEncoder  # noqa: E402
from DiffusionMRI.AxialConv2dAE import ConvDecoder  # noqa: E402

script_start = time.time()

# gpu settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # set False whenever input size varies
print("Device: ", device)

# deterministic
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False  # set False whenever input size varies
torch.backends.cudnn.deterministic = True

# training data settings
mu = 383
std = 828
batch_size = 16

data_path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/agajan/experiment_DiffusionMRI/data/train/'
trainset = Datasets.OrientationDataset(data_path + 'axial/', mu=mu, std=std, normalize=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
print("Total training examples: ", len(trainset))

# model settings
encoder = ConvEncoder(input_channels=288)
encoder.to(device)

decoder = ConvDecoder(out_channels=288)
decoder.to(device)

# load pretrained weights
prev_epoch = 0
# encoder_path = '/home/agajan/DeepMRI/DiffusionMRI/models/axial_conv2d_encoder_epoch_' + str(prev_epoch)
# decoder_path = '/home/agajan/DeepMRI/DiffusionMRI/models/axial_conv2d_decoder_epoch_' + str(prev_epoch)
# encoder.load_state_dict(torch.load(encoder_path))
# decoder.load_state_dict(torch.load(decoder_path))

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

print("Before training: ")
avg_loss, total = utils.batch_loss(trainloader, encoder, decoder, criterion, device)
ev_st = time.time()
avg_loss, total = utils.batch_loss(trainloader, encoder, decoder, criterion, device)
ev_time = time.time() - ev_st
print("Iteration: {}, Total examples: {}, Avg. loss: {}, Evaluation time: {:.5f} seconds.".format(iters,
                                                                                                  total,
                                                                                                  avg_loss,
                                                                                                  ev_time))

print("Training started for {} epochs".format(num_epochs))
for epoch in range(1, num_epochs + 1):
    running_loss = 0.0
    epoch_start = time.time()

    encoder.train()
    decoder.train()

    for data in trainloader:
        iter_time = time.time()

        optimizer.zero_grad()
        x = data.to(device)

        out = decoder(encoder(x))

        loss = criterion(x, out)
        loss.backward()
        optimizer.step()

        running_loss = running_loss + loss.item() * data.size(0)
        if iters % 100 == 0:
            print("Iter #{}, iter time: {:.5f}, batch loss: {}".format(iters, time.time() - iter_time, loss.item()))

            # report statistics for the whole set
            ev_st = time.time()
            avg_loss, total = utils.batch_loss(trainloader, encoder, decoder, criterion, device)
            ev_time = time.time() - ev_st
            print("Iteration: {}, Total examples: {}, Avg. loss: {}, Evaluation time: {:.5f}".format(iters,
                                                                                                     total,
                                                                                                     avg_loss,
                                                                                                     ev_time))
        iters += 1

    # if epoch % 1 == 0:
    #     torch.save(encoder.state_dict(), "models/axial_conv2d_encoder_epoch_{}".format(epoch + prev_epoch))
    #     torch.save(decoder.state_dict(), "models/axial_conv2d_decoder_epoch_{}".format(epoch + prev_epoch))

    epoch_loss = running_loss / len(trainset)

    # scheduler.step(epoch_loss)
    print("Epoch #{}/{},  epoch loss: {}, epoch time: {:.5f} seconds".format(epoch + prev_epoch, num_epochs, epoch_loss,
                                                                             time.time() - epoch_start))

print("Total running time: {}".format(time.time() - script_start))
