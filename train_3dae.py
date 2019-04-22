import torch
import torch.nn as nn
import datasets
from ConvAE import ConvAE
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

torch.backends.cudnn.benchmark = True  # set False whenever input size varies

batch_size = 64
trainset = datasets.Slice3dDataset('/home/agajan/tensors_3d/train/')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)
print("Total examples: ", len(trainset))

model = ConvAE()
model.to(device)
model.train()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

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

        out = model(x)
        loss = criterion(x, out)
        bp_time = time.time()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        # print("Iter #{}, iter time: {:.5f}".format(iters, time.time() - iter_time))
        iters += 1

    if epoch % 10 == 0:
        torch.save(model.state_dict(), "models/convae_epoch_{}".format(epoch))

    epoch_loss = running_loss / len(trainset)
    print("Epoch #{}|{},  epoch loss: {}, epoch time: {:.5f} seconds".format(epoch, num_epochs, epoch_loss,
                                                                             time.time() - epoch_start))
