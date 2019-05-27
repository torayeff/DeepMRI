import sys
import os

sys.path.append('/home/agajan/DeepMRI')
from deepmri import utils  # noqa: E402

data_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/'
subj_ids = ['784565', '786569', '789373']
tract_names = ['CG_left', 'CG_right', 'CST_left', 'CST_right', 'FX_left', 'FX_right', 'CC']

for subj_id in subj_ids:
    for tract_name in tract_names:
        print("Processing tract: {} of subj_id={}".format(tract_name, subj_id))
        trk_file_path = os.path.join(data_dir, subj_id, 'tracts', tract_name + '.trk')
        mask_output_path = os.path.join(data_dir, subj_id, 'tract_masks', tract_name + '_binary_mask.nii.gz')
        ref_img_path = os.path.join(data_dir, subj_id, 'nodif_brain_mask.nii.gz')

        utils.create_tract_mask(trk_file_path, mask_output_path, ref_img_path, hole_closing=2, blob_th=50)
        print("-"*100)
    print("*"*100)
