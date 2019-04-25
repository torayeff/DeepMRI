import nibabel as nib
import utils
import os
import pickle
import torch
import random
import shutil
from shutil import copyfile
from ConvEncoder import ConvEncoder
import time
from ConvAE import ConvAE

# minmin = 1000
# maxmax = 0
#
# paths = os.listdir('/home/agajan/data/')
# for path in paths:
#     x = nib.load('/home/agajan/data/' + path)
#     sh = x.shape[3]
#     print(sh)
#     if sh < minmin:
#         minmin = sh
#
#     if sh > maxmax:
#         maxmax = sh
#
# print(minmin, maxmax)


# ----------------------------------------make 4d dataset----------------------------------------
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Device: ", device)
#
# data_path = '/home/agajan/data/'
# save_path = '/home/agajan/feature_tensors_4d/'
# image_names = list(filter(lambda x: x.endswith('.nii.gz'), os.listdir(data_path)))
# image_paths = [os.path.join(data_path, img_name) for img_name in image_names]
# print('Total 4D images: ', len(image_paths))
#
# model = ConvEncoder()
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
#     src = os.path.join(data_path, f)
#     dst = os.path.join(train_path, f)
#     shutil.move(src, dst)
#
# for f in valid:
#     src = os.path.join(data_path, f)
#     dst = os.path.join(valid_path, f)
#     shutil.move(src, dst)

# ----------------------------------------make 3d dataset----------------------------------------
# # slice saver
# data_path = '/home/agajan/data/'
# save_path = '/home/agajan/tensors_3d/'
# image_names = list(filter(lambda x: x.endswith('.nii.gz'), os.listdir(data_path)))
# image_paths = [os.path.join(data_path, img_name) for img_name in image_names]
# print('Total 4D images: ', len(image_paths))
#
# count = 1
# for img_name in image_names:
#     print("{}/{}".format(count, len(image_paths)))
#     count += 1
#
#     img = nib.load(os.path.join(data_path, img_name)).get_fdata()
#
#     for i in range(img.shape[3]):
#         new_name = img_name[:-7] + '_slice_' + str(i) + '.tensor'
#         with open(os.path.join(save_path, new_name), "wb") as f:
#
#             x = img[:, :, :, i]
#             x = torch.tensor(x).unsqueeze(0).float()
#             pickle.dump(x, f)
#
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
#     src = os.path.join(data_path, f)
#     dst = os.path.join(train_path, f)
#     shutil.move(src, dst)
#
# for f in valid:
#     src = os.path.join(data_path, f)
#     dst = os.path.join(valid_path, f)
#     shutil.move(src, dst)

# ----------------------------------------make test dataset----------------------------------------
# path = "/home/agajan/Pittsburgh/"
# folders = os.listdir(path)
#
# for folder in folders:
#     if os.path.isdir(path + folder):
#         for file in os.listdir(path + folder):
#             if file.startswith("sfnwmrda") and file.endswith(".nii.gz"):
#                 print(file)
#                 copyfile(path + folder + "/" + file, "/home/agajan/test_data/" + file)
