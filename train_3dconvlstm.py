import torch
import datasets


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

batch_size = 1
trainset = datasets.FMRIDataset('/home/agajan/data/')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)

epochs = 1

for epoch in epochs:
    for data in trainloader:
        pass
