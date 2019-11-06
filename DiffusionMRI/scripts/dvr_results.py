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
# FEATURES_NAME = "PCA"
FEATURES_NAME = "GROUND_TRUTH"
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
class_colors = np.array([
    [1, 255, 255, 0],  # cyan for other class
    [255, 1, 1, 255],  # red for CG class
    [1, 255, 1, 255],  # green for CST class
    [1, 255, 255, 255],  # blue for FX class
    [255, 255, 1, 255],  # yellow for CC class
])

colors = np.array([
    [1, 255, 255],  # cyan for other class
    [255, 1, 1],  # red for CG class
    [1, 255, 1],  # green for CST class
    [1, 255, 255],  # blue for FX class
    [255, 255, 1],  # yellow for CC class
]).T  # 3x5

opacities = np.array([0, 255, 255, 255, 255])
theta = 0.0

import time
st = time.time()
# for each voxel position
for crd_idx, crd in enumerate(test_coords):
    probs = test_probs[:, crd_idx]
    probs = probs - theta
    probs[probs < 0] = 0
    probs = probs / (1 - theta)

    # opacity
    # A_sum, R_sum, G_sum, B_sum = 0, 0, 0, 0
    # RGBA = np.array([0, 0, 0, 0])
    # for c in range(5):
    #     A_sum += probs[c] * class_colors[c][3]
    #     R_sum += probs[c] * class_colors[c][0] * class_colors[c][3]
    #     G_sum += probs[c] * class_colors[c][1] * class_colors[c][3]
    #     B_sum += probs[c] * class_colors[c][2] * class_colors[c][3]
    #
    # R = 0 if A_sum == 0 else R_sum / A_sum
    # G = 0 if A_sum == 0 else G_sum / A_sum
    # B = 0 if A_sum == 0 else B_sum / A_sum
    # A = min(255, A_sum)
    # data_matrix[crd[0], crd[1], crd[2]] = [R, G, B, A]

    A_sum = np.sum(opacities * probs)
    if A_sum != 0:
        RGB = np.sum(probs * colors * opacities, axis=1) / A_sum
    else:
        RGB = np.array([0, 0, 0])
    A = min(255, A_sum)

    data_matrix[crd[0], crd[1], crd[2]] = [*RGB, A]


print("{:.5f}".format(time.time() - st))
print(np.sum(data_matrix))
visutils.dvr_rgba(data_matrix.astype(np.uint8), (1.0, 1.0, 1.0))
