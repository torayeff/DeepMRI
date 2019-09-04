from os.path import join
import numpy as np
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# settings
subj_id = '789373'
exp_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/'
data_path = join(exp_dir, subj_id, 'data.nii.gz')
mask_path = join(exp_dir, subj_id, 'nodif_brain_mask.nii.gz')

nc = 22  # number of components

save_path = join(exp_dir, subj_id, 'voxels_pca_nc_{}.npz'.format(nc))

print("Loading data.")
data = nib.load(data_path).get_data()
mask = nib.load(mask_path).get_data()

print("Making data matrix")
coords = []
features = []
for x in range(145):
    for y in range(174):
        for z in range(145):
            if mask[x, y, z]:
                coords.append((x, y, z))
                features.append(data[x, y, z, :])

print("Normalizing.")
features = StandardScaler().fit_transform(features)

print("Performing PCA.")
pca = PCA(n_components=nc, random_state=0)
features_reduced = pca.fit_transform(features)

print("Making features volume.")
features_volume = np.zeros((145, 174, 145, nc))
for idx, crd in enumerate(coords):
    features_volume[crd[0], crd[1], crd[2], :] = features_reduced[idx]

np.savez(save_path, data=features_volume)

