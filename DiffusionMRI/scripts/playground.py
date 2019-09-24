import sys
from os.path import join
import numpy as np
import nibabel as nib
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics

sys.path.append('/home/agajan/DeepMRI')
from deepmri import dsutils  # noqa: E402

SUBJ_ID = "784565"
print("SUBJECT ID={}".format(SUBJ_ID).center(100, "-"))

# ----------------------------------------------Settings----------------------------------------------

DATA_DIR = "/home/agajan/experiment_DiffusionMRI/tractseg_data/"
TRACT_MASKS_PTH = join(DATA_DIR, SUBJ_ID, "tract_masks", "tract_masks.nii.gz")

# FEATURES_NAME = "RAW"
# FEATURES_FILE = "data.nii.gz"

# FEATURES_NAME = "SHORE4
# FEATURES_FILE = "shore_features/shore_coefficients_radial_border_4.npz"

FEATURES_NAME = "PCA"
FEATURES_FILE = "unnorm_voxels_pca_nc_10.npz"

# FEATURES_NAME = "MSCONVAE_exp"
# FEATURES_FILE = "learned_features/MultiScale_features_epoch_10.npz"

FEATURES_NAME = "MSCONVAE_exp"
FEATURES_FILE = "learned_features/Model1_linear_h10_features_epoch_10.npz"

FEATURES_PATH = join(DATA_DIR, SUBJ_ID, FEATURES_FILE)
LABELS = ["Other", "CG", "CST", "FX", "CC"]

MIN_SAMPLES_LEAF = 8
ADD_COORDS = False
if ADD_COORDS:
    FEATURES_NAME = FEATURES_NAME + "_COORDS"
RESULTS_PATH = join(DATA_DIR, SUBJ_ID, "outputs", FEATURES_NAME + "_dice_scores.npz")

# ---------------------------------------------Load Data----------------------------------------------

print("Loading Data".center(100, "-"))

TRACT_MASKS = nib.load(TRACT_MASKS_PTH).get_data()
TRACT_MASKS = TRACT_MASKS[:, :, :, 1:]  # remove background class

if FEATURES_PATH.endswith(".npz"):
    FEATURES = np.load(FEATURES_PATH)["data"]
else:
    FEATURES = nib.load(FEATURES_PATH).get_data()

print(FEATURES.sum())
print(FEATURES[72, 87, 72, :])
