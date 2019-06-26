from os.path import join
import numpy as np
import nibabel as nib

# settings
subj_id = '784565'
exp_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/'
data_path = join(exp_dir, subj_id, 'data.nii.gz')
mask_path = join(exp_dir, subj_id, 'nodif_brain_mask.nii.gz')
fbval = join(exp_dir, subj_id, 'bvals')
fbvec = join(exp_dir, subj_id, 'bvecs')

nh = 3
data = np.load(join(exp_dir, subj_id, 'learned_features/strided_coronal_features_epoch_400.npz'))['data']
print(data.shape)
data = data.transpose(1, 0, 2, 3)
save_path = join(exp_dir, subj_id, 'learned_features', 'avg_strided_coronal_epoch_400_nh{}.npz'.format(nh))


def get_borders(x, border, nh=3):

    start = max(x - (nh // 2), 0)
    end = min(x + (nh // 2) + 1, border)

    return start, end


averaged_features = np.zeros(data.shape)
nh = 3
print('Neighborhood: {}'.format(nh))
for x in range(145):
    bx = get_borders(x, 145)
    for y in range(174):
        by = get_borders(y, 174)
        for z in range(145):
            bz = get_borders(z, 145, nh=nh)
            averaged_features[x, y, z] = data[bx[0]:bx[1],
                                              by[0]:by[1],
                                              bz[0]:bz[1]].reshape(-1, data.shape[-1]).mean(axis=0)

np.savez(save_path, data=averaged_features)
print(averaged_features.shape)
