import torch
import torch.nn as nn
from deepmri import Datasets, utils
from deepmri.RNNEncoder import RNNEncoder
from deepmri.RNNDecoder import RNNDecoder
from deepmri.Conv3DRNNCell import Conv3DGRUCell
import time

# pytorch settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False  # set False whenever input size varies
torch.backends.cudnn.deterministic = True

# data settings
batch_size = 1
trainset = Datasets.MRIDataset('/home/agajan/experiment_DiffusionMRI/data/', seq_idxs=(0, 3))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)
print("Total training examples: ", len(trainset))

# RNN Encoder Decoder Model
encoder = RNNEncoder(
    Conv3DGRUCell,
    input_channels=1,
    hidden_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    hidden_kernel_size=3
)
encoder.to(device)
encoder.train()

decoder = RNNDecoder(
    Conv3DGRUCell,
    input_channels=1,
    hidden_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    hidden_kernel_size=3,
    output_channels=1,
    output_kernel_size=3
)
decoder.to(device)
decoder.train()

utils.count_model_parameters(encoder)
utils.count_model_parameters(decoder)

# encoder.load_state_dict(torch.load('models/final_adam_rnn_encoder_401'))
# decoder.load_state_dict(torch.load('models/final_adam_rnn_decoder_401'))

# training settings
criterion = nn.MSELoss()
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.5)
num_epochs = 10
iters = 1

print("Training started for {} epochs".format(num_epochs))

for epoch in range(1, num_epochs + 1):
    running_loss = 0.0
    epoch_start = time.time()

    for data in trainloader:
        iter_time = time.time()

        # -------------------Seq2Seq Start------------------ #
        optimizer.zero_grad()

        print(data.shape)
        src_batch = data.to(device)
        seq_len = src_batch.shape[1]

        print("rnn encoding")
        context_batch = encoder(src_batch)

        hidden_batch = context_batch

        # first input is <sos> in learning phrase representation
        # in this case it is tensor of zeros
        input_batch = src_batch.new_zeros(src_batch[:, 0, :, :, :, :].shape)

        print("rnn decoding")
        loss = src_batch.new_zeros(1)
        for t in range(seq_len):
            input_batch, hidden_batch = decoder(input_batch, hidden_batch, context_batch)
            loss += criterion(src_batch[:, t, :, :, :, :], input_batch)

        print("calculating loss")
        loss /= seq_len
        print(loss)

        print("backprop")
        loss.backward()

        print("update parameters")
        optimizer.step()
        # -------------------Seq2Seq End------------------- #

        running_loss += loss.item() * data.size(0)
        # print("Iter #{}, iter time: {:.5f}, batch loss: {}".format(iters, time.time() - iter_time, loss.item()))
        iters += 1
    # scheduler.step(running_loss)

    # if epoch % 1 == 0:
    #     torch.save(encoder.state_dict(), "models/adam_rnn_encoder_epoch_{}".format(epoch + 401))
    #     torch.save(decoder.state_dict(), "models/adam_rnn_decoder_epoch_{}".format(epoch + 401))

    epoch_loss = running_loss / len(trainset)
    print("Epoch #{}/{},  epoch loss: {}, epoch time: {:.5f} seconds".format(epoch, num_epochs, epoch_loss,
                                                                             time.time() - epoch_start))

# print("----------------Evaluation----------------")
# print("Statistics on train set:")
# utils.evaluate_rnn_encoder_decoder(encoder, decoder, criterion, trainloader, device)
#
# print("Statistics on validation set:")
# utils.evaluate_rnn_encoder_decoder(encoder, decoder, criterion, validloader, device)
#
# print("Statistics on test set:")
# utils.evaluate_rnn_encoder_decoder(encoder, decoder, criterion, testloader, device)
