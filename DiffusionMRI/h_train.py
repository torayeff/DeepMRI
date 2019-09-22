import sys
import time
import torch

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, CustomLosses, utils  # noqa: E402
from DiffusionMRI.bkpmodels.bkp3.ConvModel1 import Encoder, Decoder  # noqa: E402

script_start = time.time()

# ------------------------------------------Settings--------------------------------------------------------------------
experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
data_path = experiment_dir + 'tractseg_data/784565/training_slices/coronal/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
deterministic = True  # reproducibility
seed = 0  # random seed for reproducibility
if deterministic:
    torch.manual_seed(seed)
torch.backends.cudnn.benchmark = (not deterministic)  # set False whenever input size varies
torch.backends.cudnn.deterministic = deterministic

# data
# batch_size = 2 ** 8
batch_size = 1

start_epoch = 0  # for loading pretrained weights
num_epochs = 10  # number of epochs to trains
checkpoint = 10  # save model every checkpoint epoch

masked_loss = False
denoising = False
# ------------------------------------------Data------------------------------------------------------------------------

# trainset = Datasets.VoxelDataset(data_path,
#                                  file_name='data.nii.gz',
#                                  normalize=False,
#                                  scale=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
trainset = Datasets.OrientationDataset(data_path, scale=True, normalize=False, bg_zero=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
total_examples = len(trainset)
print("Total training examples: {}, Batch size: {}, Iters per epoch: {}".format(total_examples,
                                                                                batch_size,
                                                                                len(trainloader)))
# ------------------------------------------Model-----------------------------------------------------------------------
for h in range(1, 101):
    model_name = "ConvModel1_prelu_h{}".format(h)

    # model settings
    # encoder = Encoder(h=h)
    # decoder = Decoder(h=h)
    encoder = Encoder(input_size=(145, 145), h=h)
    decoder = Decoder(h)
    encoder.to(device)
    decoder.to(device)

    if start_epoch != 0:
        encoder_path = "{}saved_models/{}_encoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
        decoder_path = "{}saved_models/{}_decoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
        encoder.load_state_dict(torch.load(encoder_path))
        decoder.load_state_dict(torch.load(decoder_path))
        print("Loaded pretrained weights starting from epoch {}".format(start_epoch))

    p1 = utils.count_model_parameters(encoder)
    p2 = utils.count_model_parameters(decoder)
    print("Total parameters: {}, trainable parameters: {}".format(p1[0] + p2[0], p1[1] + p2[1]))

    # criterion = torch.nn.MSELoss()
    criterion = CustomLosses.MaskedLoss()
    masked_loss = True

    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           verbose=True,
                                                           min_lr=1e-6,
                                                           patience=5)
    # ------------------------------------------Training--------------------------------------------------------------------
    print("Training: {}".format(model_name))

    utils.evaluate_ae(encoder, decoder, criterion, device, trainloader, masked_loss=masked_loss, denoising=denoising)
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
                   scheduler=scheduler,
                   checkpoint=checkpoint,
                   print_iter=False,
                   eval_epoch=10,
                   masked_loss=masked_loss,
                   denoising=False,
                   prec=8,
                   logger=None)

    print("Total running time: {:.5f} seconds.".format(time.time() - script_start))
