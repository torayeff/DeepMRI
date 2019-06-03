import sys
sys.path.append('/home/agajan/DeepMRI')

from deepmri import ds_utils  # noqa: E402

csv_file = "/home/agajan/experiment_DiffusionMRI/tractseg_data/test.csv"
save_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/test/'
ds_utils.create_orientation_dataset(csv_file, save_dir, orients=(0, 1, 2), th_sum=0)
