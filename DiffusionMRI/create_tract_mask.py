import sys
import os
import numpy as np

sys.path.append('/home/agajan/DeepMRI')
from deepmri import utils  # noqa: E402

data_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/'
subj_ids = ['784565', '786569', '789373']
tract_names = ['CG_left', 'CG_right', 'CST_left', 'CST_right', 'FX_left', 'FX_right', 'CC']

# create binary masks from .trk files
for subj_id in subj_ids:
    for tract_name in tract_names:
        print("Processing tract: {} of subj_id={}".format(tract_name, subj_id))
        trk_file_path = os.path.join(data_dir, subj_id, 'tracts', tract_name + '.trk')
        mask_output_path = os.path.join(data_dir, subj_id, 'tract_masks', tract_name + '_binary_mask.nii.gz')
        ref_img_path = os.path.join(data_dir, subj_id, 'nodif_brain_mask.nii.gz')

        utils.create_tract_mask(trk_file_path, mask_output_path, ref_img_path, hole_closing=2, blob_th=50)
        print("-"*100)
    print("*"*100)

# create multilabel binary mask
labels = ['background'] + tract_names
for subj_id in subj_ids:
    print("Creating multilabel binary mask for subj id: {}".format(subj_id))
    masks_path = os.path.join(data_dir, subj_id, 'tract_masks')
    mask_ml = utils.create_multilabel_mask(labels, masks_path, vol_size=(145, 174, 145))
    save_path = os.path.join(masks_path, 'multi_label_mask.npz')
    np.savez(save_path, data=mask_ml)
