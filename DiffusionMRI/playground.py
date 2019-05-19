import sys
sys.path.append('/home/agajan/DeepMRI')

import nibabel as nib
import os
import pickle
import torch
import random
import shutil
from shutil import copyfile
import time
import pandas as pd
import h5py
import numpy as np

from deepmri import Datasets, utils
from ADHD.ConvEncoder import ConvEncoder
from ADHD.ConvAE import ConvAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # set False whenever input size varies
print("Device: ", device)
# -------------------------------Create dataset----------------------------------------------
# csv_file = "train.csv"
# save_dir = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/agajan/experiment_DiffusionMRI/data/train/'
# utils.create_orientation_dataset(csv_file, save_dir, orients=(False, False, True))
# ------------------------------Maks csv file------------------------------------------------
# path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/regina/'
# dmris = []
# with open("test.csv", "w") as f:
#     f.write("subj_id,dmri_path\n")
#     c = 0
#     for file in os.listdir(path):
#         if len(file) == 6:
#             subj_id = file
#             dmri_path = os.path.join(path, subj_id, 'Diffusion/data.nii.gz')
#             dmri = nib.load(dmri_path)
#             if c <= 5:
#                 c += 1
#                 continue
#             if dmri.shape[3] == 288:
#                 line = "{},{}\n".format(subj_id, dmri_path)
#                 f.write(line)
#                 c += 1
#             if c == 7:
#                 break

# --------------------------------HDF5 vs Pickle---------------------------------------
# data_path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/agajan/experiment_DiffusionMRI/'
#
# # copy files
# files = [f for f in sorted(os.listdir(os.path.join(data_path, 'tensors_3d/')))[:100] if f.endswith('.3dtensor')]

# for i, f in enumerate(files, 1):
#     src = os.path.join(data_path, 'tensors_3d', f)
#     dst = os.path.join(data_path, 'hdf5_vs_pickle/tensors_3d', f)
#     shutil.copyfile(src, dst)
#     print(i)

# create hdf5 dataset with 100 files
# h5_path = os.path.join(data_path, 'hdf5_vs_pickle/volumes_hdf5/volumes3d.hdf5')
#
# with h5py.File(h5_path, 'w') as f:
#     group = f.create_group('data')
#     for idx, img_name in enumerate(files):
#         print(idx)
#
#         with open(os.path.join(data_path, 'tensors_3d', img_name), "rb") as img:
#             x = pickle.load(img).numpy()  # channel=1 x w x h x
#
#         group.create_dataset(str(idx), data=x)
#
#     group.attrs['total_data'] = len(files)

# with h5py.File(h5_path, 'r') as f:
#     data = f.get('data')
#     x = data['78']
#     x = np.array(x)
#     print(type(x))
# -------------------------------Calculate Mean and Std of Trainset--------------------
data_path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/agajan/experiment_DiffusionMRI/' \
            'data/train/axial/'
trainset = Datasets.OrientationDataset(data_path, normalize=False)

total_n = 14298 * 288 * 145 * 174  # known in advance
mu, std, n = utils.pooled_mean_std(trainset, total_n)
print(mu, std, n)

# verify
# print("Verification")
# start = time.time()
# x = trainset[0]
#
# N = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
# c = 1
# for c, data in enumerate(trainset, 1):
#     if c == 1:
#         continue
#     x = torch.cat((x, data))
#     N += data.shape[0] * data.shape[1] * data.shape[2] * data.shape[3]
#     if c == 20:
#         break
# print(x.shape)
# print("Mean: {}, Std: {}, N: {}".format(x.mean(), x.std(), N))
# print("Time: {}".format(time.time() - start))
# -------------------------------Count Dimensions--------------------------------------
# path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/regina/'
# dims = []
# c = 1
# folders = os.listdir(path)
# for file in folders:
#     if len(file) == 6:
#         subj_id = file
#         src = os.path.join(path, subj_id, 'Diffusion/data.nii.gz')
#         print(c)
#         c += 1
#
#         dim = nib.load(src).shape
#         dims.append(dim)
#
# counts = {}
# for dim in dims:
#     if dim in counts.keys():
#         counts[dim] += 1
#     else:
#         counts[dim] = 1
#
# print(counts)

# ----------------------------------------copy data----------------------------------------------
# path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/'
# c = 1
# folders = os.listdir(path)
# for file in folders:
#     if len(file) == 6:
#         subj_id = file
#         src = os.path.join(path, subj_id, 'regina', 'Diffusion/data.nii.gz')
#         dst = os.path.join(path, agajan/experiment_DiffusionMRI/data/data_{}.nii.gz'.format(subj_id))
#         print(c)
#         c += 1
#         shutil.copyfile(src, dst)
#         if c == 32:
#             break

# ----------------------------------------make 4d dataset----------------------------------------
# mu = 307.3646240234375
# std = 763.4876098632812
#
# data_path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/agajan/experiment_DiffusionMRI/data/'
# save_path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/agajan/experiment_DiffusionMRI/tensors_4d/'
# image_names = list(filter(lambda x: x.endswith('.nii.gz'), os.listdir(data_path)))
# image_paths = [os.path.join(data_path, img_name) for img_name in image_names]
# print('Total 4D images: ', len(image_paths))
#
# encoder = ConvEncoder(1)
# encoder.to(device)
# encoder.load_state_dict(torch.load('/home/agajan/DeepMRI/DiffusionMRI/models/conv_encoder_epoch_21'))
# encoder.eval()
#
# count = 1
# total = len(image_paths)
# with torch.no_grad():
#
#     for image_name in image_names:
#         print(image_name)
#         st = time.time()
#
#         img_4d = nib.load(os.path.join(data_path, image_name))
#         print(img_4d.shape)
#         img_4d = img_4d.get_fdata()
#
#         out_4d = torch.zeros(img_4d.shape[3], 64, 19, 22, 19)
#
#         for i in range(img_4d.shape[3]):
#             tensor_3d = torch.tensor(img_4d[:, :, :, i]).unsqueeze(0).unsqueeze(0).float()
#             tensor_3d = tensor_3d.to(device)
#
#             # !!!!!!!!!!!!!!! IMPORTANT
#             tensor_3d = (tensor_3d - mu) / std
#
#             # print(tensor_3d.min(), tensor_3d.max(), tensor_3d.mean(), tensor_3d.std())
#             feature = encoder(tensor_3d).cpu().squeeze()
#             # print(feature.min(), feature.max(), feature.mean(), feature.std())
#             out_4d[i, :, :, :, :] = feature
#
#         print("{}/{} time: {:.5f}".format(count, total, time.time() - st))
#         count += 1
#         new_path = os.path.join(save_path, image_name[:-7] + '.4dtensor')
#         # print(new_path)
#         # print(out_4d.shape)
#         with open(new_path, "wb") as f:
#             pickle.dump(out_4d, f)

# split train and valid
# data_path = '/home/agajan/feature_tensors_4d/'
# train_path = '/home/agajan/feature_tensors_4d/train/'
# valid_path = '/home/agajan/feature_tensors_4d/valid/'
#
# all_files = []
# for f in os.listdir(data_path):
#     if f.endswith('.4dtensor'):
#         all_files.append(f)
#
# print(all_files[:2])
# random.shuffle(all_files)
#
# cutoff = int(len(all_files) * 0.9)
# train = all_files[:cutoff]
# valid = all_files[cutoff:]
#
# for f in train:
#     deepmri = os.path.join(data_path, f)
#     dst = os.path.join(train_path, f)
#     shutil.move(deepmri, dst)
#
# for f in valid:
#     deepmri = os.path.join(data_path, f)
#     dst = os.path.join(valid_path, f)
#     shutil.move(deepmri, dst)

# ----------------------------------------make 3d dataset----------------------------------------
# data_path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/regina/'
# save_path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/agajan/experiment_DiffusionMRI/tensors_3d/'
# subj_ids = list(filter(lambda x: len(x) == 6, os.listdir(data_path)))
# image_paths = [os.path.join(data_path, subj_id, 'Diffusion') for subj_id in subj_ids]
# print('Total 4D images: ', len(image_paths))
#
# count = 1
# for subj_id in subj_ids:
#     print("{}/{}".format(count, len(subj_ids)))
#     count += 1
#
#     img = nib.load(os.path.join(data_path, subj_id, 'Diffusion/data.nii.gz')).get_fdata()
#
#     for i in range(img.shape[3]):
#         new_name = 'data_' + subj_id + '_volume_' + str(i) + '.3dtensor'
#         with open(os.path.join(save_path, new_name), "wb") as f:
#
#             x = img[:, :, :, i]
#             x = torch.tensor(x).unsqueeze(0).float()  # add channel
#             pickle.dump(x, f)

# # split train and valid
# data_path = '/home/agajan/tensors_3d/'
# train_path = '/home/agajan/tensors_3d/train/'
# valid_path = '/home/agajan/tensors_3d/valid/'
#
# all_files = []
# for f in os.listdir(data_path):
#     if f.endswith('.tensor'):
#         all_files.append(f)
#
# print(all_files[:2])
# random.shuffle(all_files)
#
# cutoff = int(len(all_files) * 0.9)
# train = all_files[:cutoff]
# valid = all_files[cutoff:]
#
# for f in train:
#     deepmri = os.path.join(data_path, f)
#     dst = os.path.join(train_path, f)
#     shutil.move(deepmri, dst)
#
# for f in valid:
#     deepmri = os.path.join(data_path, f)
#     dst = os.path.join(valid_path, f)
#     shutil.move(deepmri, dst)
