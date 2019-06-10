import sys
import os
import numpy as np
import nibabel as nib

sys.path.append('/home/agajan/DeepMRI')
from deepmri import ds_utils  # noqa: E402

data_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/'
subj_ids = ['784565', '786569', '789373']
tract_names = ['CG_left', 'CG_right', 'CST_left', 'CST_right', 'FX_left', 'FX_right', 'CC']

# create binary masks from .trk files
# for subj_id in subj_ids:
#     for tract_name in tract_names:
#         print("Processing tract: {} of subj_id={}".format(tract_name, subj_id))
#         trk_file_path = os.path.join(data_dir, subj_id, 'tracts', tract_name + '.trk')
#         mask_output_path = os.path.join(data_dir, subj_id, 'tract_masks', tract_name + '_binary_mask.nii.gz')
#         ref_img_path = os.path.join(data_dir, subj_id, 'nodif_brain_mask.nii.gz')
#
#         ds_utils.create_tract_mask(trk_file_path, mask_output_path, ref_img_path, hole_closing=2, blob_th=50)
#         print("-"*100)
#     print("*"*100)

# create multilabel binary mask
labels = ['background', 'other'] + tract_names
for subj_id in subj_ids:
    print("Creating multilabel binary mask for subj id: {}".format(subj_id))
    ref_img_path = os.path.join(data_dir, subj_id, 'nodif_brain_mask.nii.gz')
    ref_img = nib.load(ref_img_path)
    ref_affine = ref_img.affine

    masks_path = os.path.join(data_dir, subj_id, 'tract_masks')
    nodif_brain_mask_path = os.path.join(data_dir, subj_id, 'nodif_brain_mask.nii.gz')
    mask_ml = ds_utils.create_multilabel_mask(labels, masks_path, nodif_brain_mask_path, vol_size=(145, 174, 145))
    save_path = os.path.join(masks_path, 'left_right_multi_label_mask.npz')
    np.savez(save_path, data=mask_ml)
    nib.save(nib.Nifti1Image(mask_ml.astype("uint8"), ref_affine),
             os.path.join(masks_path, 'left_right_multi_label_mask.nii.gz'))

    # save also merged
    merged_ml_masks = np.zeros((145, 174, 145, 6))

    # background
    merged_ml_masks[:, :, :, 0] = mask_ml[:, :, :, 0]

    # other class
    merged_ml_masks[:, :, :, 1] = mask_ml[:, :, :, 1]

    # CG_left + CG_right
    merged_ml_masks[:, :, :, 2] = mask_ml[:, :, :, 2] + mask_ml[:, :, :, 3]

    # CST_left + CST_right
    merged_ml_masks[:, :, :, 3] = mask_ml[:, :, :, 4] + mask_ml[:, :, :, 5]

    # FX_left + FX_right
    merged_ml_masks[:, :, :, 4] = mask_ml[:, :, :, 6] + mask_ml[:, :, :, 7]

    # CC
    merged_ml_masks[:, :, :, 5] = mask_ml[:, :, :, 8]

    # remove additions
    merged_ml_masks = merged_ml_masks.clip(max=1)

    save_path = os.path.join(masks_path, 'multi_label_mask.npz')
    np.savez(save_path, data=merged_ml_masks.astype('uint8'))
    nib.save(nib.Nifti1Image(merged_ml_masks.astype("uint8"), ref_affine),
             os.path.join(masks_path, 'multi_label_mask.nii.gz'))
