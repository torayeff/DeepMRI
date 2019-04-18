import torch
import datasets
import utils
import time
from Encoder import Encoder
from Conv3DRNNCell import Conv3DGRUCell, Conv3DLSTMCell
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

batch_size = 1
trainset = datasets.FMRIDataset('/home/agajan/data/')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

epochs = 1
iters = 1

# for 3d conv seq2seq: spatial dimensions of input
# must be same with spatial dimensions of hidden

model = Encoder(
    Conv3DGRUCell,
    input_channels=1,
    hidden_channels=3,
    kernel_size=1,
    stride=1,
    padding=1,
    hidden_kernel_size=3
)

model.to(device)

for epoch in range(epochs):
    for data in trainloader:
        print(data.shape)

        start = time.time()
        print("Iteration #{} has started".format(iters))
        iters += 1

        out = model(data.to(device))
        print(out[0].shape)

        break
