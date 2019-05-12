import sys
import time
import torch
import torch.nn as nn
import nibabel as nib
import os

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, utils  # noqa: E402
from Compression.ConvEncoder import ConvEncoder  # noqa: E402
from Compression.ConvDecoder import ConvDecoder  # noqa: E402

script_start = time.time()

# gpu settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # set False whenever input size varies
print("Device: ", device)

# training data settings
batch_size = 1
data_path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/agajan/experiment_Compression/data/'
# trainset = Datasets.MRIDataset(data_path, normalize=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
# print("Total training examples: ", len(trainset))
data = nib.load(os.path.join(data_path, 'sub_01_data.nii.gz')).get_fdata()
data = torch.tensor(data).unsqueeze(0).unsqueeze(0).float()
data = (data - data.mean()) / data.std()
data = data.to(device)
trainloader = [data]

# model settings
encoder = ConvEncoder(input_channels=1)
encoder.to(device)
encoder.train()

decoder = ConvDecoder(out_channels=1)
decoder.to(device)
decoder.train()

# load pretrained weights
prev_epoch = 0
# encoder_path = '/home/agajan/DeepMRI/Compression/models/encoder_epoch_' + str(prev_epoch)
# decoder_path = '/home/agajan/DeepMRI/Compression/models/decoder_epoch_' + str(prev_epoch)
# encoder.load_state_dict(torch.load(encoder_path))
# decoder.load_state_dict(torch.load(decoder_path))

p1 = utils.count_model_parameters(encoder)
p2 = utils.count_model_parameters(decoder)
print("Total parameters: {}, trainable parameters: {}".format(p1[0] + p2[0], p1[1] + p2[1]))

# criterion and optimizer settings
criterion = nn.MSELoss()
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.5)

# training
num_epochs = 10000
iters = 1
print("Training started for {} epochs".format(num_epochs))
for epoch in range(1, num_epochs + 1):
    running_loss = 0.0
    epoch_start = time.time()

    for data in trainloader:
        iter_time = time.time()

        optimizer.zero_grad()
        # x = data.to(device)
        x = data

        out = decoder(encoder(x))

        loss = criterion(x, out)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        # if iters % 100 == 0:
        #     print("Iter #{}, iter time: {:.5f}, batch loss: {}".format(iters, time.time() - iter_time, loss.item()))

            # torch.save(encoder.state_dict(), "models/encoder_epoch_{}_iter_{}"
            #            .format(epoch + prev_epoch, iters))
            # torch.save(decoder.state_dict(), "models/decoder_epoch_{}_iter_{}"
            #            .format(epoch + prev_epoch, iters))
        iters += 1

    if epoch % 1000 == 0:
        torch.save(encoder.state_dict(), "models/encoder_epoch_{}".format(epoch + prev_epoch))
        torch.save(decoder.state_dict(), "models/decoder_epoch_{}".format(epoch + prev_epoch))

    # epoch_loss = running_loss / len(trainset)
    epoch_loss = running_loss

    # scheduler.step(epoch_loss)
    print("Epoch #{}/{},  epoch loss: {}, epoch time: {:.5f} seconds".format(epoch + prev_epoch, num_epochs, epoch_loss,
                                                                             time.time() - epoch_start))

print("Total running time: {}".format(time.time() - script_start))
