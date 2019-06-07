import sys
import time
import torch
import torch.nn as nn

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, CustomLosses, utils  # noqa: E402
from DiffusionMRI.Conv2dAESagittal import ConvEncoder, ConvDecoder  # noqa: E402

experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
data_path = experiment_dir + 'tractseg_data/train/sagittal/'
model_name = "SagittalConv2dAE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
batch_size = 11

trainset = Datasets.OrientationDatasetChannelNorm(data_path, normalize=True, bg_zero=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=10)
print(len(trainset))

encoder = ConvEncoder(input_channels=288)
decoder = ConvDecoder(out_channels=288)
encoder.to(device)
decoder.to(device)

criterion = nn.MSELoss(reduction='sum')
masked_mse = CustomLosses.MaskedMSE()
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.0003)

for b, batch in enumerate(trainloader):
    st = time.time()
    optimizer.zero_grad()
    bs = batch['data'].shape[0]
    x = batch['data'].to(device)
    out = decoder(encoder(x))

    mask = batch['mask'].unsqueeze(1).to(device)
    x = torch.mul(x, mask)
    out = torch.mul(out, mask)

    # loss = criterion(x, out) / torch.sum(mask)
    # loss = masked_mse(x, out, mask)
    loss = 0
    ch = batch['data'].shape[1]

    for j in range(bs):
        sample_loss = criterion(x[j], out[j]) / (mask[j].sum())
        loss = loss + sample_loss
    loss = loss / bs

    loss.backward()
    optimizer.step()
    print("{:.5f}".format(time.time() - st))

    print("Batch #: {}, batch size: {}, loss: {}".format(b, bs, loss.item()))




# batch = next(iter(trainloader))
#
# x = batch['data'].to(device)
# out = decoder(encoder(x))
#
# mask = batch['mask'].unsqueeze(1).to(device)
# x = torch.mul(x, mask)
# out = torch.mul(out, mask)
#
# loss = criterion(x, out) / torch.sum(mask)
# custom_loss = masked_mse(x, out, mask)
#
# avg_loss = 0
# bs = batch['data'].shape[0]
# ch = batch['data'].shape[1]
#
# for b in range(bs):
#     sample_loss = criterion(x[b], out[b]) / (ch * mask[b].sum())
#     avg_loss = avg_loss + sample_loss
# avg_loss = avg_loss / bs
#
# print(loss.item(), custom_loss.item(), avg_loss.item())

