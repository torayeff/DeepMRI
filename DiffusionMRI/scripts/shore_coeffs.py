import nibabel as nib
from dipy.reconst.shore import ShoreModel
from dipy.core.gradients import gradient_table
from dipy.io import read_bvals_bvecs
from os.path import join
import time
import numpy as np

# settings
subj_id = '784565'
exp_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/'
data_path = join(exp_dir, subj_id, 'data.nii.gz')
mask_path = join(exp_dir, subj_id, 'nodif_brain_mask.nii.gz')
fbval = join(exp_dir, subj_id, 'bvals')
fbvec = join(exp_dir, subj_id, 'bvecs')

# make gradient table
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

# load dmri data
print("Loading data and mask.")
data = nib.load(data_path).get_data()
print("Data shape: ", data.shape)

# load mask
mask = nib.load(mask_path).get_data()
print("Mask shape: ", mask.shape)

st = time.time()
radial_border = 6

print("Started fitting SHORE model with radial_border={}".format(radial_border))
asm = ShoreModel(gtab, radial_order=radial_border)
asmfit = asm.fit(data, mask)
shore_coeff = asmfit.shore_coeff
print("SHORE coefficients shape: ", shore_coeff.shape)

# replace nan with 0
print("Replacing with nan with 0. This happens because of inaccurate mask.")
shore_coeff = np.nan_to_num(shore_coeff, 0)

print("Fitting time: {:.5f}".format(time.time()-st))

save_path = join(exp_dir, subj_id, 'shore_features/shore_coefficients_radial_border_{}.npz'.format(radial_border))
np.savez(save_path, data=shore_coeff)
