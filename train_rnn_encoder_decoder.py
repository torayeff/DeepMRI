import torch
import torch.nn as nn
import Datasets
from RNNEncoder import RNNEncoder
from RNNDecoder import RNNDecoder
from Conv3DRNNCell import Conv3DGRUCell
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

torch.backends.cudnn.benchmark = True  # set False whenever input size varies

batch_size = 16
trainset = Datasets.Feature4dDataset('/home/agajan/feature_tensors_4d/train/', max_seq_len=50)
validset = Datasets.Feature4dDataset('/home/agajan/feature_tensors_4d/valid/', max_seq_len=50)
# trainset = Datasets.Feature4dDataset('/home/agajan/smallset/', max_seq_len=50)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=6)
print("Total training examples: ", len(trainset))
print("Total validation examples: ", len(validset))

encoder = RNNEncoder(
    Conv3DGRUCell,
    input_channels=64,
    hidden_channels=128,
    kernel_size=3,
    stride=1,
    padding=1,
    hidden_kernel_size=3
)
encoder.to(device)
encoder.train()

decoder = RNNDecoder(
    Conv3DGRUCell,
    input_channels=64,
    hidden_channels=128,
    kernel_size=3,
    stride=1,
    padding=1,
    hidden_kernel_size=3,
    output_channels=64,
    output_kernel_size=3
)
decoder.to(device)
decoder.train()

encoder.load_state_dict(torch.load('models/final_adam_rnn_encoder_199'))
decoder.load_state_dict(torch.load('models/final_adam_rnn_decoder_199'))

criterion = nn.MSELoss()
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.5)
num_epochs = 1000
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
        trg_batch = data.to(device)

        seq_len = trg_batch.shape[1]

        context_batch = encoder(src_batch)

        hidden_batch = context_batch

        # first input is <sos> in learning phrase representation
        # in this case it is tensor of ones
        input_batch = trg_batch.new_zeros(trg_batch[:, 0, :, :, :, :].shape)

        outputs = trg_batch.new_zeros(trg_batch.shape)

        for t in range(seq_len):
            input_batch, hidden_batch = decoder(input_batch, hidden_batch, context_batch)
            outputs[:, t, :, :, :, :] = input_batch

        loss = criterion(trg_batch, outputs)
        loss.backward()
        optimizer.step()
        # -------------------Seq2Seq End------------------- #

        running_loss += loss.item() * data.size(0)
        # print("Iter #{}, iter time: {:.5f}, batch loss: {}".format(iters, time.time() - iter_time, loss.item()))
        iters += 1
    scheduler.step(running_loss)

    if epoch % 1 == 0:
        torch.save(encoder.state_dict(), "models/adam_rnn_encoder_epoch_{}".format(epoch + 199))
        torch.save(decoder.state_dict(), "models/adam_rnn_decoder_epoch_{}".format(epoch + 199))

    epoch_loss = running_loss / len(trainset)
    print("Epoch #{}/{},  epoch loss: {}, epoch time: {:.5f} seconds".format(epoch, num_epochs, epoch_loss,
                                                                             time.time() - epoch_start))

# print("----------------Validation----------------")
#
# with torch.no_grad():
#     encoder.eval()
#     decoder.eval()
#     running_loss = 0.0
#     epoch_start = time.time()
#
#     for data in validloader:
#         iter_time = time.time()
#
#         # -------------------Seq2Seq Start------------------ #
#
#         src_batch = data.to(device)
#         trg_batch = data.to(device)
#
#         seq_len = trg_batch.shape[1]
#
#         context_batch = encoder(src_batch)
#
#         hidden_batch = context_batch
#
#         # first input is <sos> in learning phrase representation
#         # in this case it is tensor of ones
#         input_batch = trg_batch.new_ones(trg_batch[:, 0, :, :, :, :].shape)
#
#         outputs = trg_batch.new_zeros(trg_batch.shape)
#
#         for t in range(seq_len):
#             input_batch, hidden_batch = decoder(input_batch, hidden_batch, context_batch)
#             outputs[:, t, :, :, :, :] = input_batch
#
#         loss = criterion(trg_batch, outputs)
#         # -------------------Seq2Seq End------------------- #
#
#         running_loss += loss.item() * data.size(0)
#
#         epoch_loss = running_loss / len(trainset)
#
#     print("Average loss: {}".format(running_loss / len(validset)))
