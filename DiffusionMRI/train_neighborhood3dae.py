import sys
import torch

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, utils  # noqa: E402
from DiffusionMRI.NHConv3dAE import ConvEncoder, ConvDecoder  # noqa: E402

# ------------------------------------------Settings--------------------------------------------------------------------
experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
data_path = experiment_dir + 'tractseg_data/784565/'
model_name = "NHConv3dAE3x3x3"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
torch.backends.cudnn.benchmark = True  # set False whenever input size varies

# data
batch_size = 512

start_epoch = 0  # for loading pretrained weights
num_epochs = 1000  # number of epochs to trains
checkpoint = 100  # save model every checkpoint epoch
# ------------------------------------------Data------------------------------------------------------------------------

trainset = Datasets.NeighborhoodDataset(data_path, normalize=False)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
total_examples = len(trainset)
print("Total training examples: {}, Batch size: {}, Iters per epoch: {}".format(total_examples,
                                                                                batch_size,
                                                                                total_examples / batch_size))
# ------------------------------------------Model-----------------------------------------------------------------------
# model settings
encoder = ConvEncoder()
decoder = ConvDecoder()
encoder.to(device)
decoder.to(device)

if start_epoch != 0:
    encoder_path = "{}/models/{}_encoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
    decoder_path = "{}/models/{}_decoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    print("Loaded pretrained weights starting from epoch {}".format(start_epoch))

p1 = utils.count_model_parameters(encoder)
p2 = utils.count_model_parameters(decoder)
print("Total parameters: {}, trainable parameters: {}".format(p1[0] + p2[0], p1[1] + p2[1]))

# criterion and optimizer settings
# criterion = CustomLosses.MaskedMSE()
# masked_loss = True

criterion = torch.nn.MSELoss()
masked_loss = False

parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.0003)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       verbose=True,
                                                       min_lr=1e-6,
                                                       patience=5)
# ------------------------------------------Training--------------------------------------------------------------------
print("Training: {}".format(model_name))
utils.evaluate_ae(encoder, decoder, criterion, device, trainloader, masked_loss=masked_loss)
utils.train_ae(encoder,
               decoder,
               criterion,
               optimizer,
               device,
               trainloader,
               num_epochs,
               model_name,
               experiment_dir,
               start_epoch=start_epoch,
               scheduler=None,
               checkpoint=checkpoint,
               print_iter=False,
               eval_epoch=100,
               masked_loss=masked_loss)
