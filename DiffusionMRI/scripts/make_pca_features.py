from os.path import join
import numpy as np
import nibabel as nib
import sys

sys.path.append('/home/agajan/DeepMRI')
from deepmri import dsutils  # noqa: E402

# settings
subj_id = '784565'
exp_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/'
data_path = join(exp_dir, subj_id, 'data.nii.gz')
mask_path = join(exp_dir, subj_id, 'nodif_brain_mask.nii.gz')

nc = 10  # number of components

save_path = join(exp_dir, subj_id, 'unnorm_voxels_pca_nc_{}.npz'.format(nc))

print("Loading data.")
data = nib.load(data_path).get_data()
mask = nib.load(mask_path).get_data()

features_volume, pca = dsutils.make_pca_volume(data, mask, n_components=nc, normalize=False)

np.savez(save_path, data=features_volume)

