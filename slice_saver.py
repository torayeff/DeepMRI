import nibabel as nib
import utils
import os
import pickle
import torch

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

            # pad to 64x64x64
            x = img[:, :, :, i]
            x = utils.pad_3d(x, target_dims=(64, 64, 64))
            x = torch.tensor(x).unsqueeze(0).float()
            pickle.dump(x, f)
