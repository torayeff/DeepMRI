import sys
from os.path import join
import numpy as np

sys.path.append('/home/agajan/DeepMRI')

from deepmri import visutils  # noqa: E402

SUBJ_ID = "784565"
print("SUBJECT ID={}".format(SUBJ_ID).center(100, "-"))

# ----------------------------------------------Settings----------------------------------------------
DATA_DIR = "/home/agajan/experiment_DiffusionMRI/tractseg_data/"
ADD_COORDS = False
FEATURES_NAME = "PCA"
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

# RGBA
class_colors = [
    [0, 255, 0, 0],  # cyan for other class
    [255, 0, 0, 255],  # red for CG class
    [0, 255, 0, 255],  # green for CST class
    [0, 0, 255, 255],  # blue for FX class
    [255, 255, 0, 255],  # yellow for CC class
]
theta = 0.5

# for each voxel position
for crd_idx, crd in enumerate(test_coords):
    probs = test_probs[:, crd_idx, 1]
    probs = probs - theta
    probs[probs < 0] = 0
    probs /= (1 - theta)

    # opacity
    A_sum, R_sum, G_sum, B_sum = 0, 0, 0, 0
    for c in range(5):
        A_sum += probs[c] * class_colors[c][3]
        R_sum += probs[c] * class_colors[c][0] * class_colors[c][3]
        G_sum += probs[c] * class_colors[c][1] * class_colors[c][3]
        B_sum += probs[c] * class_colors[c][2] * class_colors[c][3]

    R = 0 if A_sum == 0 else R_sum / A_sum
    G = 0 if A_sum == 0 else G_sum / A_sum
    B = 0 if A_sum == 0 else B_sum / A_sum
    A = min(255, A_sum)

    data_matrix[crd[0], crd[1], crd[2]] = np.uint8([R, G, B, A])

data_matrix = np.uint8(data_matrix)
visutils.dvr_rgba(data_matrix, (1.0, 1.0, 1.0))
