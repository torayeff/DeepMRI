import sys
from os.path import join
import numpy as np
import time

from numpy.core._multiarray_umath import ndarray

st = time.time()
sys.path.append('/home/agajan/DeepMRI')

from deepmri import visutils  # noqa: E402

SUBJ_ID = "784565"
print("SUBJECT ID={}".format(SUBJ_ID).center(100, "-"))

# ----------------------------------------------Settings----------------------------------------------
DATA_DIR = "/home/agajan/experiment_DiffusionMRI/tractseg_data/"
ADD_COORDS = False
# FEATURES_NAME = "GROUND_TRUTH"
# FEATURES_NAME = "PCA"
FEATURES_NAME = "MSCONVAE"
LABELS = ["Other", "CG", "CST", "FX", "CC"]
if ADD_COORDS:
    FEATURES_NAME = FEATURES_NAME + "_COORDS"
PROBS_COORDS_PATH = join(DATA_DIR, SUBJ_ID, "outputs", FEATURES_NAME + "_probs_coords.npz")

probs_coords = np.load(PROBS_COORDS_PATH)
test_probs = probs_coords["probs"]
test_coords = probs_coords["coords"]

# ----------------------------------------------Blending----------------------------------------------
print("Blending".center(100, "-"))

vol_size = (145, 174, 145)
data_matrix = np.zeros((*vol_size, 4))

colors = np.array([
    [1, 255, 255],  # cyan for other class
    [255, 1, 1],  # red for CG class
    [1, 255, 1],  # green for CST class
    [1, 255, 255],  # blue for FX class
    [255, 255, 1],  # yellow for CC class
]).T

opacities = np.array([0, 255, 255, 255, 255])

theta = 0.5

# for each voxel position
for crd_idx, crd in enumerate(test_coords):
    probs = test_probs[:, crd_idx]

    # max prob
    # max_prob = np.max(probs)
    # probs[probs < max_prob] = 0

    # threshold
    probs = probs - theta
    probs[probs < 0] = 0
    probs = probs / (1 - theta)

    # vectorize
    A_sum = np.sum(opacities * probs)
    if A_sum != 0:
        RGB = np.sum(probs * colors * opacities, axis=1) / A_sum
    else:
        RGB = np.array([0, 0, 0])
    A = min(255.0, float(A_sum))

    data_matrix[crd[0], crd[1], crd[2]] = [*RGB, A]

print("Comp. time: {:.2f} seconds.".format(time.time() - st))
visutils.dvr_rgba(data_matrix.astype(np.uint8), (1.0, 1.0, 1.0))
