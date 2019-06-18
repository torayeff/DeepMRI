import sys
sys.path.append('/home/agajan/DeepMRI')

from deepmri import ds_utils  # noqa: E402

subj_id = 784565
csv_file = "/home/agajan/experiment_DiffusionMRI/tractseg_data/{}/data.csv".format(subj_id)
save_dir = "/home/agajan/experiment_DiffusionMRI/tractseg_data/{}/orientation_slices/".format(subj_id)
ds_utils.create_orientation_dataset(csv_file, save_dir, orients=(0, 2,), th_sum=-1, avg=False, within_brain=True)
