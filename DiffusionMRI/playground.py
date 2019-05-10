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

from deepmri import Datasets, utils
from ADHD.ConvEncoder import ConvEncoder
from ADHD.ConvAE import ConvAE


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Device: ", device)

# -------------------------------Calculate Mean and Std of Trainset--------------------
# data_path = '/media/schultz/345de007-c698-4c33-93c1-3964b99c5df6/agajan/experiment_DiffusionMRI/tensors_3d/'
# trainset = Datasets.Volume3dDataset(data_path)
#
# total_n = 148781436150  # known in advance
# # total_n = 73167000
# mu, std, n = utils.pooled_mean_std(trainset, total_n)
# print(mu, std, n)

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

#
# data_path = '/home/agajan/test_data/'
# save_path = '/home/agajan/test_feature_tensors_4d/'
# image_names = list(filter(lambda x: x.endswith('.nii.gz'), os.listdir(data_path)))
# image_paths = [os.path.join(data_path, img_name) for img_name in image_names]
# print('Total 4D images: ', len(image_paths))
#
# model = ConvEncoder(1)
# model.to(device)
# model.load_state_dict(torch.load('/home/agajan/DeepMRI/models/final_conv_encoder'))
# model.eval()
#
# count = 1
# total = len(image_paths)
# with torch.no_grad():
#
#     for image_name in image_names:
#         st = time.time()
#
#         img_4d = nib.load(os.path.join(data_path, image_name)).get_fdata()
#
#         out_4d = torch.zeros(img_4d.shape[3], 64, 7, 8, 6)
#
#         for i in range(img_4d.shape[3]):
#             slice_3d = torch.tensor(img_4d[:, :, :, i]).unsqueeze(0).unsqueeze(0).float()# batch x channel x w x h x d
#             feature = model(slice_3d.to(device)).cpu().squeeze()
#             out_4d[i, :, :, :, :] = feature
#
#         print("{}/{} time: {:.5f}".format(count, total, time.time() - st))
#         count += 1
#         new_path = os.path.join(save_path, image_name[:-7] + '.4dtensor')
#         # print(new_path)
#         # print(out_4d.shape)
#         with open(new_path, "wb") as f:
#             pickle.dump(out_4d, f)
#         # break

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
