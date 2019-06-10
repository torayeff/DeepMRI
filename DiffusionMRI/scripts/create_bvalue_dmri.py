import nibabel as nib
import numpy as np
from os.path import join

subj_id = '786569'
experiment_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/'
dmri_path = join(experiment_dir, subj_id, 'data.nii.gz')
dmri = nib.load(dmri_path)
dmri_data = dmri.get_data()

pth = join(experiment_dir, subj_id, 'bvals')
with open(pth, "r") as f:
    bvals = f.readlines()[0].rstrip(" \n")
    bvals = bvals.split("  ")

bvals_1 = [round(int(x)/100)*100 for x in bvals]

dmri_3000 = np.zeros((145, 174, 145, 90))

idx = 0
for t, bval in enumerate(bvals_1):
    if bval == 3000:
        dmri_3000[:, :, :, idx] = dmri_data[:, :, :, t]
        idx += 1

dmri_3000_nifti = nib.Nifti1Image(dmri_3000, dmri.affine)
nib.save(dmri_3000_nifti, join(experiment_dir, subj_id, 'data_3000.nii.gz'))
