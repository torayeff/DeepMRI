import nibabel as nib
import utils
import os
import pickle
import torch
import random
import shutil
from ConvAE import ConvAE

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Device: ", device)
#
# # make 4d dataset
# data_path = '/home/agajan/data/'
# save_path = '/home/agajan/tensors_4d/'
# image_names = list(filter(lambda x: x.endswith('.nii.gz'), os.listdir(data_path)))
# image_paths = [os.path.join(data_path, img_name) for img_name in image_names]
# print('Total 4D images: ', len(image_paths))
#
# model = ConvAE()
# model.to(device)
# model.load_state_dict(torch.load('/home/agajan/DeepMRI/models/sgd_convae_epoch_10'))
# model.eval()
#
# with torch.no_grad():
#
#     for image_path in image_paths:
#         print(image_path)
#
#         img_4d = nib.load(image_path).get_fdata()
#
#         for i in range(img_4d.shape[3]):
#             slice_3d = torch.tensor(img_4d[:, :, :, i]).unsqueeze(0).unsqueeze(0).float()  # batch x channel x w x h x d
#             feature = model(slice_3d.to(device))
#             print(feature.shape)
#             break
#
#         break


# make 3d dataset
# slice saver
data_path = '/home/agajan/data/'
save_path = '/home/agajan/tensors_3d/'
image_names = list(filter(lambda x: x.endswith('.nii.gz'), os.listdir(data_path)))
image_paths = [os.path.join(data_path, img_name) for img_name in image_names]
print('Total 4D images: ', len(image_paths))

count = 1
for img_name in image_names:
    print("{}/{}".format(count, len(image_paths)))
    count += 1

    img = nib.load(os.path.join(data_path, img_name)).get_fdata()

    for i in range(img.shape[3]):
        new_name = img_name[:-7] + '_slice_' + str(i) + '.tensor'
        with open(os.path.join(save_path, new_name), "wb") as f:

            x = img[:, :, :, i]
            x = torch.tensor(x).unsqueeze(0).float()
            pickle.dump(x, f)

# split train and valid
data_path = '/home/agajan/tensors_3d/'
train_path = '/home/agajan/tensors_3d/train/'
valid_path = '/home/agajan/tensors_3d/valid/'

all_files = []
for f in os.listdir(data_path):
    if f.endswith('.tensor'):
        all_files.append(f)

print(all_files[:2])
random.shuffle(all_files)

cutoff = int(len(all_files) * 0.9)
train = all_files[:cutoff]
valid = all_files[cutoff:]

for f in train:
    src = os.path.join(data_path, f)
    dst = os.path.join(train_path, f)
    shutil.move(src, dst)

for f in valid:
    src = os.path.join(data_path, f)
    dst = os.path.join(valid_path, f)
    shutil.move(src, dst)
