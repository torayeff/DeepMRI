import sys
import time
import torch
import wandb

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, CustomLosses, utils  # noqa: E402
from DiffusionMRI.models.ConcatModel import Encoder, Decoder  # noqa: E402  # noqa: E402

script_start = time.time()

# ------------------------------------------Settings--------------------------------------------------------------------
experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
data_path = experiment_dir + 'tractseg_data/784565/training_slices/coronal/'
model_name = "ConcatModel"
wandb.init(project="deepmri", name=model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
deterministic = True  # reproducibility
seed = 0  # random seed for reproducibility
if deterministic:
    torch.manual_seed(seed)
torch.backends.cudnn.benchmark = (not deterministic)  # set False whenever input size varies
torch.backends.cudnn.deterministic = deterministic

# data
batch_size = 1

# noise probability
noise_prob = None

start_epoch = 0  # for loading pretrained weights
num_epochs = 30  # number of epochs to trains
checkpoint = 1  # save model every checkpoint epoch
# ------------------------------------------Data------------------------------------------------------------------------

trainset = Datasets.OrientationDataset(data_path,
                                       scale=True,
                                       normalize=False,
                                       bg_zero=True,
                                       noise_prob=noise_prob,
                                       alpha=1)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=10)
print("Total training examples: {}, Batch size: {}, Iters per epoch: {}".format(len(trainset),
                                                                                batch_size,
                                                                                len(trainloader)))
# ------------------------------------------Model-----------------------------------------------------------------------
# model settings
encoder = Encoder(input_size=(145, 145))
decoder = Decoder()
encoder.to(device)
decoder.to(device)

if start_epoch != 0:
    encoder_path = "{}/saved_models/{}_encoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
    decoder_path = "{}/saved_models/{}_decoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    print("Loaded pretrained weights starting from epoch {}".format(start_epoch))

p1 = utils.count_model_parameters(encoder)
p2 = utils.count_model_parameters(decoder)
print("Total parameters: {}, trainable parameters: {}".format(p1[0] + p2[0], p1[1] + p2[1]))

# criterion and optimizer settings
criterion = CustomLosses.MaskedLoss()
masked_loss = True

parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25)
# ------------------------------------------Training--------------------------------------------------------------------
print("Training: {}".format(model_name))
utils.evaluate_ae(encoder, decoder, criterion, device, trainloader, masked_loss=masked_loss, denoising=bool(noise_prob))
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
               logger=wandb)

print("Total running time: {}".format(time.time() - script_start))
