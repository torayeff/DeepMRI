import torch
import torch.nn as nn
import Datasets
from RNNEncoder import RNNEncoder
from Conv3DRNNCell import Conv3DGRUCell
from ADHDClassifier import ADHDClassifier
import utils
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

torch.backends.cudnn.benchmark = True  # set False whenever input size varies

batch_size = 64
trainset = Datasets.ADHDFeatureDataset('/home/agajan/feature_tensors_4d/train/',
                                       csv_file='/home/agajan/DeepMRI/adhd_trainset.csv',
                                       seq_len=50,
                                       binary=True)
validset = Datasets.ADHDFeatureDataset('/home/agajan/feature_tensors_4d/valid/',
                                       csv_file='/home/agajan/DeepMRI/adhd_trainset.csv',
                                       seq_len=50,
                                       binary=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=6)

print("Total training examples: ", len(trainset))
print("Total validation examples: ", len(validset))

rnn_encoder = RNNEncoder(
    Conv3DGRUCell,
    input_channels=64,
    hidden_channels=128,
    kernel_size=3,
    stride=1,
    padding=1,
    hidden_kernel_size=3
)
rnn_encoder.to(device)
rnn_encoder.train()
rnn_encoder.load_state_dict(torch.load('models/final_adam_rnn_encoder_199'))

classifier = ADHDClassifier(128, 7, 8, 6)
classifier.to(device)
classifier.train()

for param in rnn_encoder.parameters():
    param.requires_grad = False
    rnn_encoder.eval()

parameters = list(classifier.parameters()) #+ list(rnn_encoder.parameters())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(parameters, lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.5)
num_epochs = 100
iters = 100

print("Training started for {} epochs".format(num_epochs))

for epoch in range(1, num_epochs + 1):
    running_loss = 0.0
    epoch_start = time.time()

    for x, y in trainloader:
        iter_time = time.time()

        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        features = rnn_encoder(x)
        outputs = classifier(features)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        # print("Iter #{}, iter time: {:.5f}, batch loss: {}".format(iters, time.time() - iter_time, loss.item()))
        iters += 1

    scheduler.step(running_loss)

    # if epoch % 1 == 0:
    #     torch.save(rnn_encoder.state_dict(), "models/adhd_classifier_encoder_epoch_{}".format(epoch))
    #     torch.save(classifier.state_dict(), "models/adhd_classifier_epoch_{}".format(epoch))

    epoch_loss = running_loss / len(trainset)
    print("Epoch #{}/{},  epoch loss: {}, epoch time: {:.5f} seconds".format(epoch, num_epochs, epoch_loss,
                                                                             time.time() - epoch_start))

    # evaluate after each epoch
    if epoch % 10 == 0:
        print("Statistics on test set: ")
        utils.evaluate_adhd_classifier(classifier, rnn_encoder, criterion, trainloader, device)

        print("Statistics on validation set: ")
        utils.evaluate_adhd_classifier(classifier, rnn_encoder, criterion, validloader, device)

