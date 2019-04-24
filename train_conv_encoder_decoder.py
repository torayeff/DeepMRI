import torch
import torch.nn as nn
import Datasets
from ConvEncoder import ConvEncoder
from ConvDecoder import ConvDecoder
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

torch.backends.cudnn.benchmark = True  # set False whenever input size varies

batch_size = 256
trainset = Datasets.Slice3dDataset('/home/agajan/tensors_3d/train/')
validset = Datasets.Slice3dDataset('/home/agajan/tensors_3d/valid/')
# trainset = Datasets.Slice3dDataset('/home/agajan/smallset/')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=6)
print("Total training examples: ", len(trainset))
print("Total validation examples: ", len(validset))

encoder = ConvEncoder()
encoder.to(device)
encoder.train()

decoder = ConvDecoder()
decoder.to(device)
decoder.train()

criterion = nn.MSELoss()
# parameters = list(encoder.parameters()) + list(decoder.parameters())

# optimizer = torch.optim.Adam(parameters)
# encoder.load_state_dict(torch.load('models/final_conv_encoder'))
# decoder.load_state_dict(torch.load('models/final_conv_decoder'))
# optimizer = torch.optim.SGD(parameters, lr=0.00001, momentum=0.9)
#
# num_epochs = 100
# iters = 1
#
# print("Training started for {} epochs".format(num_epochs))
#
# for epoch in range(1, num_epochs + 1):
#     running_loss = 0.0
#     epoch_start = time.time()
#
#     for data in trainloader:
#         iter_time = time.time()
#
#         optimizer.zero_grad()
#         x = data.to(device)
#
#         out = decoder(encoder(x))
#
#         loss = criterion(x, out)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item() * data.size(0)
#         # print("Iter #{}, iter time: {:.5f}, batch loss: {}".format(iters, time.time() - iter_time, loss.item()))
#         iters += 1
#
#     if epoch % 1 == 0:
#         torch.save(encoder.state_dict(), "models/temp_conv_encoder_epoch_{}".format(epoch))
#         torch.save(decoder.state_dict(), "models/temp_conv_decoder_epoch_{}".format(epoch))
#
#     epoch_loss = running_loss / len(trainset)
#     print("Epoch #{}/{},  epoch loss: {}, epoch time: {:.5f} seconds".format(epoch, num_epochs, epoch_loss,
#                                                                              time.time() - epoch_start))

print("----------------Validation----------------")
encoder.load_state_dict(torch.load('models/final_conv_encoder'))
decoder.load_state_dict(torch.load('models/final_conv_decoder'))

with torch.no_grad():
    running_loss = 0.0
    encoder.eval()
    decoder.eval()
    for data in validloader:
        x = data.to(device)
        out = decoder(encoder(x))
        loss = criterion(x, out)
        print("batch loss: {}".format(loss.item()))
        running_loss += loss.item() * data.size(0)
    print("Average loss: {}".format(running_loss / len(validset)))
