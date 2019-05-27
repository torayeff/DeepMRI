import sys
import os

sys.path.append('/home/agajan/DeepMRI')
from deepmri import utils  # noqa: E402

data_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/'
subj_id = '784565'
tract_name = 'CC'

trk_file_path = os.path.join(data_dir, subj_id, 'tracts', tract_name + '.trk')
mask_output_path = os.path.join(data_dir, subj_id, 'tract_masks', tract_name + '_binary_mask.nii.gz')
ref_img_path = os.path.join(data_dir, subj_id, 'nodif_brain_mask.nii.gz')

utils.create_tract_mask(trk_file_path, mask_output_path, ref_img_path, hole_closing=2, blob_th=50)
