import sys
sys.path.append('/home/agajan/DeepMRI')

from deepmri import ds_utils  # noqa: E402

subj_id = 786569
csv_file = "/home/agajan/experiment_DiffusionMRI/tractseg_data/{}/data.csv".format(subj_id)
save_dir = "/home/agajan/experiment_DiffusionMRI/tractseg_data/{}/orientation_slices/".format(subj_id)
ds_utils.create_orientation_dataset(csv_file, save_dir, orients=(1, ), th_sum=0, within_brain=True)
