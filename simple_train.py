import torch
import torch.nn as nn
import nibabel as nib
import utils
import time
import pickle
from ConvAE import ConvAE


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

torch.backends.cudnn.benchmark = True  # set False whenever input size varies

# anat = '/home/agajan/mylab/fmridata/KKI/1018959/wssd1018959_session_1_anat.nii.gz'
# img = nib.load(anat)
# img_data = img.get_fdata()
# print("Image shape: ", img_data.shape)
img_data = nib.load('/home/agajan/mylab/fmridata/KKI/1018959/sfnwmrda1018959_session_1_rest_1.nii.gz').get_fdata()[:, :, :, 0]

# pad
# img_data = utils.pad_3d(img_data)
# print("Padded image shape: ", img_data.shape)

# convert to batch tensor
x_batch = torch.tensor(img_data).float().unsqueeze(0).unsqueeze(0)  # batch x C x D x H x W
print(torch.max(x_batch), torch.min(x_batch))
print(x_batch.shape)

x_batch = x_batch.to(device)
print("X batch shape: ", x_batch.shape)

model = ConvAE()
model.to(device)
model.train()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

iters = 1000

for it in range(1, iters + 1):
    tic = time.time()

    optimizer.zero_grad()
    out = model(x_batch)
    loss = criterion(x_batch, out)
    loss.backward()
    optimizer.step()

    toc = time.time()
    print("Iteration #{}, Loss: {}, Time: {:.5f}".format(it, loss.item(), toc - tic))

    if (it + 1) % 10 == 0:
        torch.save(model.state_dict(), "models/iter_{}".format(it))
