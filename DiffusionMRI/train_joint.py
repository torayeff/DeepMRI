import torch
import torch.nn as nn
import time
import sys

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, utils  # noqa: E402
from DiffusionMRI.ConvEncoder import ConvEncoder  # noqa: E402
from DiffusionMRI.ConvDecoder import ConvDecoder  # noqa: E402
from deepmri.RNNEncoder import RNNEncoder  # noqa: E402
from deepmri.RNNDecoder import RNNDecoder  # noqa: E402
from deepmri.Conv3DRNNCell import Conv3DGRUCell  # noqa: E402


# pytorch settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True  # set False whenever input size varies
# torch.backends.cudnn.deterministic = True

# data settings
batch_size = 1
data_path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/agajan/experiment_DiffusionMRI/'

# trainset = Datasets.MRIDataset(data_path + 'data/one_data/', seq_idxs=(0, 10))
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
# print("Total training examples: ", len(trainset))

import nibabel as nib
import os
print("Loading data")
st = time.time()
data = nib.load(os.path.join(data_path, 'data/one_data/data_151526.nii.gz')).get_data()
seq_len = 10
data = data.transpose(3, 0, 1, 2)  # time x width x height x depth
data = torch.tensor(data).float()[0:seq_len]
data = data.unsqueeze(1).unsqueeze(0)  # batch x time x channel x width x height x depth

trainloader = [data]
trainset = [data]
print("Loaded data: ", time.time() - st)

# model settings
conv_encoder = ConvEncoder(input_channels=1)
conv_decoder = ConvDecoder(out_channels=1)

rnn_encoder = RNNEncoder(
    Conv3DGRUCell,
    input_channels=64,
    hidden_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    hidden_kernel_size=3
)

rnn_decoder = RNNDecoder(
    Conv3DGRUCell,
    input_channels=64,
    hidden_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    hidden_kernel_size=3,
    output_channels=64,
    output_kernel_size=3
)

# send models to device
conv_encoder.to(device)
conv_decoder.to(device)
rnn_encoder.to(device)
rnn_decoder.to(device)

# count parameters
p1 = utils.count_model_parameters(conv_encoder)
p2 = utils.count_model_parameters(conv_decoder)
p3 = utils.count_model_parameters(rnn_encoder)
p4 = utils.count_model_parameters(rnn_decoder)
print("Total trainable parameters: {}".format(p1[0] + p2[0] + p3[0] + p4[0]))

prev_epoch = 0
# encoder.load_state_dict(torch.load('models/rnn_encoder_{}.format(prev_epoch)'))
# decoder.load_state_dict(torch.load('models/rnn_decoder_{}'.format(prev_epoch)))

# criterion and optimizer settings
ConvLoss = nn.MSELoss()
RNNLoss = nn.MSELoss()

parameters = list(conv_encoder.parameters()) + \
             list(conv_decoder.parameters()) + \
             list(rnn_encoder.parameters()) + \
             list(rnn_decoder.parameters())

optimizer = torch.optim.Adam(parameters, lr=0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.5)

# training
conv_encoder.train()
conv_decoder.train()
rnn_encoder.train()
rnn_decoder.train()

num_epochs = 100000
iters = 1

mu = data.mean()
std = data.std()

print("Training started for {} epochs".format(num_epochs))
for epoch in range(1, num_epochs + 1):
    running_loss = 0.0
    epoch_start = time.time()

    for data in trainloader:
        iter_time = time.time()

        optimizer.zero_grad()

        # pass 3D volume through ConvAE
        conv_loss = torch.tensor([0.], requires_grad=True, device=device)
        features = []
        fshape = None

        for t in range(data.size(1)):
            x = data[:, t, :, :, :, :].to(device)
            x = (x - mu) / std
            feature = conv_encoder(data[:, t, :, :, :, :].to(device))
            fshape = feature.shape
            x_out = conv_decoder(feature)
            conv_loss = conv_loss + ConvLoss(x, x_out)
            features.append(feature.unsqueeze(0))  # channel x w x h x d

        # create src_batch from features
        src_batch = torch.zeros(1, seq_len, fshape[1], fshape[2], fshape[3], fshape[4])
        for t in range(seq_len):
            src_batch[:, t, :, :, :, :] = features[t]

        # -------------------Seq2Seq Start------------------ #
        src_batch = src_batch.to(device)

        context_batch = rnn_encoder(src_batch)

        hidden_batch = context_batch

        # first input is <sos> in learning phrase representation
        # in this case it is tensor of zeros
        input_batch = src_batch.new_zeros(src_batch[:, 0, :, :, :, :].shape)

        rnn_loss = src_batch.new_zeros(1)
        for t in range(seq_len):
            input_batch, hidden_batch = rnn_decoder(input_batch, hidden_batch, context_batch)
            rnn_loss = rnn_loss + RNNLoss(src_batch[:, t, :, :, :, :], input_batch)

        rnn_loss = rnn_loss / seq_len
        # -------------------Seq2Seq End------------------- #

        combined_loss = conv_loss + rnn_loss
        running_loss = running_loss + combined_loss.item()
        # print("Combined Loss: {}".format(combined_loss.item()))

        # backprop
        st = time.time()
        combined_loss.backward()

        # update parameters
        st = time.time()
        optimizer.step()

    epoch_loss = running_loss / len(trainset)
    print("Epoch #{}/{},  epoch loss: {}, epoch time: {:.5f} seconds".format(epoch, num_epochs, epoch_loss,
                                                                             time.time() - epoch_start))

    if epoch % 1000 == 0:
        torch.save(conv_encoder.state_dict(), "models/overfit_joint_conv_encoder_epoch_{}".format(epoch + prev_epoch))
        torch.save(conv_decoder.state_dict(), "models/overfit_joint_conv_decoder_epoch_{}".format(epoch + prev_epoch))
        torch.save(rnn_encoder.state_dict(), "models/overfit_joint_rnn_encoder_epoch_{}".format(epoch + prev_epoch))
        torch.save(rnn_decoder.state_dict(), "models/overfit_joint_rnn_decoder_epoch_{}".format(epoch + prev_epoch))
