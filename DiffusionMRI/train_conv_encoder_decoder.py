import sys
sys.path.append('/home/agajan/DeepMRI')
import torch
import torch.nn as nn
from deepmri import Datasets, utils
from DiffusionMRI.ConvEncoder import ConvEncoder
from DiffusionMRI.ConvDecoder import ConvDecoder
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

torch.backends.cudnn.benchmark = True  # set False whenever input size varies

batch_size = 16
data_path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/agajan/experiment_DiffusionMRI/'
trainset = Datasets.Slice3dDataset(data_path + 'tensors_3d/')
# trainset = Datasets.Slice3dDataset('/home/agajan/experiment_DiffusionMRI/smallset/train/')
# validset = Datasets.Slice3dDataset('/home/agajan/experiment_DiffusionMRI/feature_tensors_3d/valid/')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)
# validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=6)
print("Total training examples: ", len(trainset))
# print("Total validation examples: ", len(validset))

encoder = ConvEncoder(input_channels=1)
encoder.to(device)
encoder.train()

decoder = ConvDecoder(out_channels=1)
decoder.to(device)
decoder.train()

criterion = nn.MSELoss()
parameters = list(encoder.parameters()) + list(decoder.parameters())

# encoder_path = '/home/agajan/DeepMRI/DiffusionMRI/models/conv_encoder_epoch_' + str(10000)
# decoder_path = '/home/agajan/DeepMRI/DiffusionMRI/models/conv_decoder_epoch_' + str(10000)
# encoder.load_state_dict(torch.load(encoder_path))
# decoder.load_state_dict(torch.load(decoder_path))
optimizer = torch.optim.Adam(parameters)
# optimizer = torch.optim.SGD(parameters, lr=0.0005, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.5)
# utils.count_model_parameters(encoder)
# utils.count_model_parameters(decoder)

num_epochs = 1000
iters = 1

print("Training started for {} epochs".format(num_epochs))

for epoch in range(1, num_epochs + 1):
    running_loss = 0.0
    epoch_start = time.time()

    for data in trainloader:
        iter_time = time.time()

        optimizer.zero_grad()
        x = data.to(device)

        out = decoder(encoder(x))

        loss = criterion(x, out)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        if iters % 100 == 0:
            print("Iter #{}, iter time: {:.5f}, batch loss: {}".format(iters, time.time() - iter_time, loss.item()))
        iters += 1

    if epoch % 1 == 0:
        torch.save(encoder.state_dict(), "models/conv_encoder_epoch_{}".format(epoch))
        torch.save(decoder.state_dict(), "models/conv_decoder_epoch_{}".format(epoch))

    epoch_loss = running_loss / len(trainset)
    # scheduler.step(epoch_loss)
    print("Epoch #{}/{},  epoch loss: {}, epoch time: {:.5f} seconds".format(epoch, num_epochs, epoch_loss,
                                                                             time.time() - epoch_start))

# print("----------------Validation----------------")
# encoder.load_state_dict(torch.load('models/final_conv_encoder'))
# decoder.load_state_dict(torch.load('models/final_conv_decoder'))
#
# with torch.no_grad():
#     running_loss = 0.0
#     encoder.eval()
#     decoder.eval()
#     for data in validloader:
#         x = data.to(device)
#         out = decoder(encoder(x))
#         loss = criterion(x, out)
#         print("batch loss: {}".format(loss.item()))
#         running_loss += loss.item() * data.size(0)
#     print("Average loss: {}".format(running_loss / len(validset)))
