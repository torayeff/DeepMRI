import sys
import torch
import torch.nn as nn
import os

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, utils  # noqa: E402
from DiffusionMRI.SagittalConv2dAE import ConvEncoder as SagittalEncoder  # noqa: E402
from DiffusionMRI.SagittalConv2dAE import ConvDecoder as SagittalDecoder  # noqa: E402

experiment_dir = '/home/agajan/experiment_DiffusionMRI/'

# gpu settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # set False whenever input size varies
print("Device: ", device)

# training data settings
mu = 368.62549
std = 823.93335
batch_size = 16

TB_DATA = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/agajan/experiment_DiffusionMRI/'
train_path = os.path.join(experiment_dir, 'data/train/sagittal_part1')
valid_path = os.path.join(TB_DATA, 'data/train/sagittal_part2')
test_path = os.path.join(TB_DATA, 'data/test/sagittal')

trainset = Datasets.OrientationDataset(train_path, mu=mu, std=std, normalize=True)
validset = Datasets.OrientationDataset(valid_path, mu=mu, std=std, normalize=True)
testset = Datasets.OrientationDataset(test_path, mu=mu, std=std, normalize=True)
print("Total training examples: ", len(trainset))
print("Total validation examples: ", len(validset))
print("Total test examples: ", len(testset))

criterion = nn.MSELoss()

# Conv2dAE
sagittal_encoder = SagittalEncoder(input_channels=288)
sagittal_encoder.to(device)

sagittal_decoder = SagittalDecoder(out_channels=288)
sagittal_decoder.to(device)

epoch = 50
sagittal_encoder.load_state_dict(torch.load(experiment_dir +
                                            'models/SagittalConv2dAE_encoder_epoch_{}'.format(epoch)))
sagittal_decoder.load_state_dict(torch.load(experiment_dir +
                                            'models/SagittalConv2dAE_decoder_epoch_{}'.format(epoch)))

print("Evaluating at epoch: {}".format(epoch))
# performance on trainset
print("Evaluation on train set".center(80, '-'))
utils.dataset_performance(trainset,
                          sagittal_encoder,
                          sagittal_decoder,
                          criterion,
                          device,
                          mu,
                          std,
                          every_iter=1000)

# performance on validset
print("Evaluation on validset set".center(80, '-'))
utils.dataset_performance(validset,
                          sagittal_encoder,
                          sagittal_decoder,
                          criterion,
                          device,
                          mu,
                          std,
                          every_iter=1000)

# performance on testset
print("Evaluation on test set".center(80, '-'))
utils.dataset_performance(testset,
                          sagittal_encoder,
                          sagittal_decoder,
                          criterion,
                          device,
                          mu,
                          std,
                          every_iter=500)
