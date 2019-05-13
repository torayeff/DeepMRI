import torch
import torch.nn as nn
import time
import sys

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, utils  # noqa: E402
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

trainset = Datasets.Feature4dDataset(data_path + 'tensors_4d/', max_seq_len=100)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
print("Total training examples: ", len(trainset))

# model settings
encoder = RNNEncoder(
    Conv3DGRUCell,
    input_channels=64,
    hidden_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    hidden_kernel_size=3
)
encoder.to(device)

decoder = RNNDecoder(
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
decoder.to(device)

p1 = utils.count_model_parameters(encoder)
p2 = utils.count_model_parameters(decoder)
print("Total parameters: {}, trainable parameters: {}".format(p1[0] + p2[0], p1[1] + p2[1]))

prev_epoch = 0
# encoder.load_state_dict(torch.load('models/rnn_encoder_{}.format(prev_epoch)'))
# decoder.load_state_dict(torch.load('models/rnn_decoder_{}'.format(prev_epoch)))

# criterion and optimizer settings
criterion = nn.MSELoss()
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.5)

# training
decoder.train()
encoder.train()

num_epochs = 10000
iters = 1

print("Training started for {} epochs".format(num_epochs))
for epoch in range(1, num_epochs + 1):
    running_loss = 0.0
    epoch_start = time.time()

    for data in trainloader:
        iter_time = time.time()

        # -------------------Seq2Seq Start------------------ #
        optimizer.zero_grad()

        src_batch = data.to(device)
        seq_len = src_batch.shape[1]

        context_batch = encoder(src_batch)

        hidden_batch = context_batch

        # first input is <sos> in learning phrase representation
        # in this case it is tensor of zeros
        input_batch = src_batch.new_zeros(src_batch[:, 0, :, :, :, :].shape)

        loss = src_batch.new_zeros(1)
        for t in range(seq_len):
            input_batch, hidden_batch = decoder(input_batch, hidden_batch, context_batch)
            loss = loss + criterion(src_batch[:, t, :, :, :, :], input_batch)

            # torch.cuda.empty_cache()

        loss = loss / seq_len

        loss.backward()

        optimizer.step()
        # -------------------Seq2Seq End------------------- #

        running_loss += loss.item() * data.size(0)
        # print("Iter #{}, iter time: {:.5f}, batch loss: {}".format(iters, time.time() - iter_time, loss.item()))
        iters += 1

        # torch.cuda.empty_cache()
    scheduler.step(running_loss)

    if epoch % 100 == 0:
        torch.save(encoder.state_dict(), "models/overfit_rnn_encoder_epoch_{}".format(epoch + prev_epoch))
        torch.save(decoder.state_dict(), "models/overfit_rnn_decoder_epoch_{}".format(epoch + prev_epoch))

    epoch_loss = running_loss / len(trainset)
    print("Epoch #{}/{},  epoch loss: {}, epoch time: {:.5f} seconds".format(epoch, num_epochs, epoch_loss,
                                                                             time.time() - epoch_start))