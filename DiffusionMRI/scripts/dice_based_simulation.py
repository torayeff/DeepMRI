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

NUM_ITERS = 10
MIN_SAMPLES_LEAF = 8
ADD_COORDS = True
if ADD_COORDS:
    FEATURES_NAME = FEATURES_NAME + "_COORDS"
SAVE_PATH = join(DATA_DIR, SUBJ_ID, "simulation_data", FEATURES_NAME + "_dice_based.npz")
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
test_sets = {}
results = {}
orients_num_slices = {
    "axial": 145,
    "coronal": 174,
    "sagittal": 145
}

orient_ranges = {
    "axial": range(52, 93),
    "coronal": range(67, 108),
    "sagittal": range(52, 93)
}

for orient, num_slices in orients_num_slices.items():
    for i in orient_ranges[orient]:
        mask_name = (orient, i)
        tmsk = dsutils.create_data_masks(TRACT_MASKS, [mask_name], LABELS, verbose=False)

        X_test, y_test, test_coords = dsutils.create_dataset_from_data_mask(FEATURES,
                                                                            tmsk,
                                                                            labels=LABELS,
                                                                            multi_label=True)
        if X_test.shape[0] == 0:
            continue
        if ADD_COORDS:
            X_test = np.concatenate((X_test, test_coords), axis=1)

        test_sets[mask_name] = {}
        test_sets[mask_name]["x"] = X_test
        test_sets[mask_name]["y"] = y_test
        test_sets[mask_name]["coords"] = test_coords
        results[mask_name] = 1

full_X_test, full_y_test, full_test_coords = dsutils.create_dataset_from_data_mask(FEATURES,
                                                                                   TRACT_MASKS.copy(),
                                                                                   labels=LABELS,
                                                                                   multi_label=True)
if ADD_COORDS:
    full_X_test = np.concatenate((full_X_test, full_test_coords), axis=1)
# ------------------------------------DICE score based simulation-------------------------------------
scores = []
tract_scores = {
    "CG": [],
    "CST": [],
    "FX": [],
    "CC": []
}
train_slices = [('sagittal', 72), ('coronal', 87), ('axial', 72)]
# remove from dictionary
for slc in train_slices:
    if slc in results.keys():
        del results[slc]
        del test_sets[slc]

for its in range(NUM_ITERS):
    iter_start = time()
    print("Simulation iteration: {}/{}".format(its + 1, NUM_ITERS).center(100, "-"))
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
    for mask_name in test_sets.keys():
        X_test, y_test, test_coords = test_sets[mask_name]["x"], test_sets[mask_name]["y"], \
                                      test_sets[mask_name]["coords"]
        if ADD_COORDS:
            X_train = np.concatenate((X_test, test_coords), axis=1)
        test_preds = clf.predict(X_test)
        test_f1_macro = sklearn.metrics.f1_score(y_test[:, 1:], test_preds[:, 1:], average='macro')

        results[mask_name] = test_f1_macro

    # Evaluate on full brain
    test_preds = clf.predict(full_X_test)

    pred_masks = dsutils.preds_to_data_mask(test_preds, full_test_coords, LABELS)
    dsutils.save_pred_masks(pred_masks, DATA_DIR, SUBJ_ID, str(its) + "_" + FEATURES_NAME)

    test_f1_macro = sklearn.metrics.f1_score(full_y_test[:, 1:], test_preds[:, 1:], average='macro')
    scores.append(test_f1_macro)
    print("Full brain F1_macro: {:.5f}".format(test_f1_macro))
    test_f1s = sklearn.metrics.f1_score(full_y_test[:, 1:], test_preds[:, 1:], average=None)

    for c, f1 in enumerate(test_f1s):
        print("{}: {:.5f}".format(LABELS[c + 1], f1), end=" ")
        tract_scores[LABELS[c + 1]].append(f1)
    print()

    sorted_results = sorted(results.items(), key=lambda kv: kv[1])
    extend_list = []
    extend_orients = []
    for res in sorted_results:
        if res[0][0] not in extend_orients:
            extend_orients.append(res[0][0])
            mask_name = res[0]
            train_slices.append(mask_name)
            del results[mask_name]
            del test_sets[mask_name]
            extend_list.append(mask_name)

        if len(extend_list) == 3:
            break

    print("Extending the training set with: ", [s for s in extend_list])
    print("Iteration time: {:.5f}".format(time() - iter_start))

np.savez(SAVE_PATH, iters=list(range(1, NUM_ITERS+1)),
         scores=scores,
         cg=tract_scores["CG"],
         cst=tract_scores["CST"],
         fx=tract_scores["FX"],
         cc=tract_scores["CC"])
print("Simulation took {:.5f} seconds".format(time() - script_start))
