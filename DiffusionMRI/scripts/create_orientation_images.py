import sys
sys.path.append('/home/agajan/DeepMRI')

from deepmri import dsutils  # noqa: E402

subj_id = 784565
csv_file = "/home/agajan/experiment_DiffusionMRI/tractseg_data/{}/shore_data.csv".format(subj_id)
save_dir = "/home/agajan/experiment_DiffusionMRI/tractseg_data/{}/shore_slices/".format(subj_id)
dsutils.create_orientation_dataset(csv_file, save_dir, orients=(1,), th_sum=0, avg=False, within_brain=True)
