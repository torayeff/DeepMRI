import sys
sys.path.append('/home/agajan/DeepMRI')

from deepmri import ds_utils  # noqa: E402

csv_file = "/home/agajan/experiment_DiffusionMRI/tractseg_data/train.csv"
save_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/784565/orients/'
ds_utils.create_orientation_dataset(csv_file, save_dir, orients=(0, 1, 2), th_sum=-1)
