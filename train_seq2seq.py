import torch
import torch.nn as nn
import datasets
import time
from Encoder import Encoder
from Decoder import Decoder
from Seq2Seq import Seq2Seq
from Conv3DRNNCell import Conv3DGRUCell
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

batch_size = 1
trainset = datasets.FMRIDataset('/home/agajan/data1/')
# trainset = datasets.FMRIDataset('/home/user/torayev/data1/')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

# for 3d conv seq2seq: spatial dimensions of input
# must be same with spatial dimensions of hidden

encoder = Encoder(
    Conv3DGRUCell,
    input_channels=1,
    hidden_channels=1,
    kernel_size=3,
    stride=1,
    padding=1,
    hidden_kernel_size=3
)

decoder = Decoder(
    Conv3DGRUCell,
    input_channels=1,
    hidden_channels=1,
    kernel_size=3,
    stride=1,
    padding=1,
    hidden_kernel_size=3,
    output_channels=1,
    output_kernel_size=3
)

# training
model = Seq2Seq(encoder, decoder)
model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=1e-2)
# optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
criterion = nn.MSELoss()

epochs = 1000
iters = 1

print("Training started for {} epochs".format(epochs))

for epoch in range(1, epochs + 1):
    epoch_loss = 0
    epoch_start = time.time()

    for data in trainloader:
        iter_time = time.time()

        optimizer.zero_grad()
        src = data.to(device)
        trg = data.to(device)

        out = model(src, trg)
        loss = criterion(trg, out)
        bp_time = time.time()
        loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        # torch.cuda.empty_cache()

        epoch_loss += loss.item()
        print("Iter #{}, iter time: {:.5f}".format(iters, time.time() - iter_time))
        iters += 1

    if epoch % 100 == 0:
        torch.save(model.state_dict(), "models/epochr_{}".format(epoch))

    print("Epoch #{},  loss: {}, epoch time: {:.5f} seconds".format(epoch, epoch_loss / len(trainset),
                                                                    time.time() - epoch_start))
