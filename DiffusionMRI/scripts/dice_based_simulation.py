import sys
from os.path import join
import numpy as np
import nibabel as nib
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
from time import time
import warnings
warnings.filterwarnings("ignore")

sys.path.append('/home/agajan/DeepMRI')
from deepmri import dsutils  # noqa: E402

script_start = time()
SUBJ_ID = "784565"
print("DICE score based simulation".center(100, "-"))
print("SUBJECT ID={}".format(SUBJ_ID).center(100, "-"))

# ----------------------------------------------Settings----------------------------------------------
DATA_DIR = "/home/agajan/experiment_DiffusionMRI/tractseg_data/"
TRACT_MASKS_PTH = join(DATA_DIR, SUBJ_ID, "tract_masks", "tract_masks.nii.gz")

# FEATURES_NAME = "RAW"
# FEATURES_FILE = "data.nii.gz"

# FEATURES_NAME = "SHORE4
# FEATURES_FILE = "shore_features/shore_coefficients_radial_border_4.npz"

# FEATURES_NAME = "PCA"
# FEATURES_FILE = "unnorm_voxels_pca_nc_10.npz"

FEATURES_NAME = "MSCONVAE"
FEATURES_FILE = "learned_features/MultiScale_features_epoch_10.npz"

FEATURES_PATH = join(DATA_DIR, SUBJ_ID, FEATURES_FILE)
LABELS = ["Other", "CG", "CST", "FX", "CC"]

SAVE_PATH = join(DATA_DIR, SUBJ_ID, "simulation_data", FEATURES_NAME + "_dice_based.npz")

NUM_ITERS = 15
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

print("FEATURES File: {}".format(FEATURES_FILE))
print("FEATURES Name: {}, shape: {}".format(FEATURES_NAME, FEATURES.shape))

# ----------------------------------------------Test Set----------------------------------------------
print("Preparing test sets.".center(100, "-"))
mask_names = []
test_sets = []
results = {}
orients_num_slices = {
    # "axial": 145,
    # "coronal": 174,
    "sagittal": 145
}

for orient, num_slices in orients_num_slices.items():
    for i in range(60, 80):
        mask_name = (orient, i)
        tmsk = dsutils.create_data_masks(TRACT_MASKS, [mask_name], LABELS, verbose=False)
        mask_names.append(mask_name)

        X_test, y_test, test_coords = dsutils.create_dataset_from_data_mask(FEATURES,
                                                                            tmsk,
                                                                            labels=LABELS,
                                                                            multi_label=True)
        if X_test.shape[0] == 0:
            continue
        if ADD_COORDS:
            X_test = np.concatenate((X_test, test_coords), axis=1)

        test_sets.append((X_test, y_test))
        results[mask_name] = 1

full_X_test, full_y_test, full_test_coords = dsutils.create_dataset_from_data_mask(FEATURES,
                                                                                   TRACT_MASKS.copy(),
                                                                                   labels=LABELS,
                                                                                   multi_label=True)
if ADD_COORDS:
    X_test = np.concatenate((full_X_test, full_test_coords), axis=1)
# ------------------------------------DICE score based simulation-------------------------------------
scores = []
train_slices = [('sagittal', 72), ('coronal', 87), ('axial', 72)]
for its in range(NUM_ITERS):
    iter_start = time()
    print("Simulation iteration: {}/{}".format(its, NUM_ITERS).center(100, "-"))
    # ---------------------------------------------Train Set----------------------------------------------
    train_masks = dsutils.create_data_masks(TRACT_MASKS, train_slices, LABELS, verbose=False)
    X_train, y_train, train_coords = dsutils.create_dataset_from_data_mask(FEATURES,
                                                                           train_masks,
                                                                           labels=LABELS,
                                                                           multi_label=True)
    if ADD_COORDS:
        X_train = np.concatenate((X_train, train_coords), axis=1)
    print("X_train shape: {}, y_train shape: {}".format(X_train.shape, y_train.shape))

    clf = RandomForestClassifier(n_estimators=100,
                                 bootstrap=True,
                                 oob_score=True,
                                 random_state=0,
                                 n_jobs=-1,
                                 max_features='auto',
                                 class_weight='balanced',
                                 max_depth=None,
                                 min_samples_leaf=MIN_SAMPLES_LEAF)
    clf.fit(X_train, y_train)
    for idx, test_set in enumerate(test_sets):
        X_test, y_test = test_set

        # ---------------------------------------Evaluation on test set---------------------------------------
        test_preds = clf.predict(X_test)
        test_f1_macro = sklearn.metrics.f1_score(y_test[:, 1:], test_preds[:, 1:], average='macro')
        results[mask_names[idx]] = test_f1_macro

    # Evaluate on full brain
    test_preds = clf.predict(full_X_test)
    test_f1_macro = sklearn.metrics.f1_score(full_y_test[:, 1:], test_preds[:, 1:], average='macro')
    scores.append(test_f1_macro)
    print("Full brain F1_macro: {:.5f}".format(test_f1_macro))

    sorted_results = sorted(results.items(), key=lambda kv: kv[1])[:3]
    for j in range(3):
        train_slices.append(sorted_results[j][0])
        del results[sorted_results[j][0]]

    print("Extending the training set with: ", [s[0] for s in sorted_results[:3]])
    print("Iteration time: {:.5f}".format(time() - iter_start))

np.savez(SAVE_PATH, iters=list(range(1, NUM_ITERS)), scores=scores)
print("Simulation took {:.5f} seconds".format(time() - script_start))
