import sys
from os.path import join

sys.path.append('/home/agajan/DeepMRI')
from deepmri import dsutils  # noqa: E402

data_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/'
subj_id = "784565"

data_pth = join(data_dir, subj_id, "tract_masks/tract_masks.nii.gz")
save_dir = join(data_dir, subj_id, "tract_masks/single_labels/")
# data_pth = join(data_dir, subj_id, "pred_masks/MSCONVAE_retrained_COORDS_pred_masks.nii.gz")
# save_dir = join(data_dir, subj_id, "pred_masks/MSCONVAE_COORDS/")

labels = ["BG", "Other", "CG", "CST", "FX", "CC"]
for idx, label in enumerate(labels):
    save_path = join(save_dir, label + ".nii.gz")
    dsutils.save_one_volume(data_pth, save_path, idx, binary=False, midx=idx-1)

dsutils.save_as_one_mask(data_pth, join(save_dir, "itkmasks.nii.gz"))