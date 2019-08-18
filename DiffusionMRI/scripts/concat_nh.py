from os.path import join
import numpy as np
import nibabel as nib

# settings
subj_id = '784565'
exp_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/'
data_path = join(exp_dir, subj_id, 'shore_features/shore_coefficients_radial_border_4.npz')
mask_path = join(exp_dir, subj_id, 'nodif_brain_mask.nii.gz')
nh = 3
save_path = join(exp_dir, subj_id, 'shore_features/shore4_nh_{}.npz'.format(nh))

print("Loading data.")
data = np.load(data_path)['data']
mask = nib.load(mask_path).get_data()

nh_features = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[3] * nh * nh * nh))
data = np.pad(data, 1, mode="constant", constant_values=0)


def get_borders(x, border, nh=3):
    start = max(x - (nh // 2), 0)
    end = min(x + (nh // 2) + 1, border)

    return start, end


print("Making neighborhood features.")
for x in range(1, 145):
    for y in range(1, 174):
        for z in range(1, 145):
            if mask[x, y, z]:
                bx = get_borders(x, 145, nh=nh)
                by = get_borders(y, 174, nh=nh)
                bz = get_borders(z, 145, nh=nh)
                fvec = data[bx[0]:bx[1], by[0]:by[1], bz[0]:bz[1], 1:23]
                nh_features[x, y, z] = fvec.reshape(-1)

np.savez(save_path, data=nh_features)