import sys
import numpy as np
import os
sys.path.append('/home/agajan/DeepMRI')

from deepmri import ds_utils  # noqa: E402

csv_file = "/home/agajan/experiment_DiffusionMRI/tractseg_data/test.csv"
save_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/test/'
ds_utils.create_orientation_dataset(csv_file, save_dir, orients=(0, 1, 2), th_sum=-1)

# make empty volumes
# empty_sagittal = np.zeros((288, 174, 145))
# np.savez(os.path.join(save_dir, 'sagittal/empty_sagittal.npz'), data=empty_sagittal)
#
# empty_coronal = np.zeros((288, 145, 145))
# np.savez(os.path.join(save_dir + 'coronal/empty_coronal.npz'), data=empty_coronal)
#
# empty_axial = np.zeros((288, 145, 174))
# np.savez(os.path.join(save_dir + 'axial/empty_axial.npz'), data=empty_axial)
