import sys
from os.path import join
import numpy as np
import nibabel as nib
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
from time import time

sys.path.append('/home/agajan/DeepMRI')
from deepmri import dsutils  # noqa: E402

script_start = time()
SUBJ_ID = "789373"
print("SUBJECT ID={}".format(SUBJ_ID).center(100, "-"))

# ----------------------------------------------Settings----------------------------------------------

DATA_DIR = "/home/agajan/experiment_DiffusionMRI/tractseg_data/"
TRACT_MASKS_PTH = join(DATA_DIR, SUBJ_ID, "tract_masks", "tract_masks.nii.gz")

FEATURES_NAMES = ["RAW288", "RAW288_COORDS",
                  "SHORE4", "SHORE4_COORDS",
                  "PCA22", "PCA22_COORDS",
                  "MODEL10"]
FEATURES_FILES = ["data.nii.gz", "data.nii.gz",
                  "shore_features/shore_coefficients_radial_border_4.npz",
                  "shore_features/shore_coefficients_radial_border_4.npz",
                  "voxels_pca_nc_22.npz", "voxels_pca_nc_22.npz",
                  "learned_features/Model10_features_epoch_200.npz"
                  ]
MIN_SAMPLES_LEAFS = [9, 9, 10, 9, 10, 9, 3]
ADD_COORDS_VALUES = [False, True, False, True, False, True, False]
NUM_ITERS = 15
FULL_BRAIN = True

for idx, FEATURES_FILE in enumerate(FEATURES_FILES):
    FEATURES_PATH = join(DATA_DIR, SUBJ_ID, FEATURES_FILE)
    FEATURES_NAME = FEATURES_NAMES[idx]
    ADD_COORDS = ADD_COORDS_VALUES[idx]
    MIN_SAMPLES_LEAF = MIN_SAMPLES_LEAFS[idx]

    LABELS = ["Other", "CG", "CST", "FX", "CC"]
    save_path = join(DATA_DIR, SUBJ_ID, "simulation_data", FEATURES_NAME + ".npz")
    print("Simulation has started for {} iterations.".format(NUM_ITERS).center(100, "-"))
    # ---------------------------------------------Load Data----------------------------------------------

    print("Loading Data".center(100, "-"))

    TRACT_MASKS = nib.load(TRACT_MASKS_PTH).get_data()
    TRACT_MASKS = TRACT_MASKS[:, :, :, 1:]  # remove background class

    if FEATURES_PATH.endswith(".npz"):
        FEATURES = np.load(FEATURES_PATH)["data"]
    else:
        FEATURES = nib.load(FEATURES_PATH).get_data()

    print("FEATURES Name: {}, shape: {}".format(FEATURES_NAME, FEATURES.shape))

    # ----------------------------------------------Test Set----------------------------------------------

    print('Preparing the test set'.center(100, '-'))
    if FULL_BRAIN:
        test_masks = TRACT_MASKS.copy()
    else:
        test_slices = [('sagittal', 71), ('coronal', 86), ('axial', 71)]
        test_masks = dsutils.create_data_masks(TRACT_MASKS, test_slices, LABELS)
    X_test, y_test, test_coords = dsutils.create_dataset_from_data_mask(FEATURES,
                                                                        test_masks,
                                                                        labels=LABELS,
                                                                        multi_label=True)
    if ADD_COORDS:
        X_test = np.concatenate((X_test, test_coords), axis=1)
    print("X_test shape: {}, y_test shape: {}".format(X_test.shape, y_test.shape))

    # ---------------------------------------------Simulation---------------------------------------------
    seed_slices = [('sagittal', 72), ('coronal', 87), ('axial', 72)]

    train_slices = []
    c = 0
    stats = {
        "iters": [],
        "scores": [],
        "train_voxels": [],
        "train_times": []
    }
    for it in range(NUM_ITERS):
        print("Simulation iter #{}".format(it + 1).center(100, "-"))
        train_start = time()
        c, train_slices = dsutils.make_training_slices(seed_slices, it, c, train_slices)
        train_masks = dsutils.create_data_masks(TRACT_MASKS, train_slices, LABELS)

        X_train, y_train, train_coords = dsutils.create_dataset_from_data_mask(FEATURES,
                                                                               train_masks,
                                                                               labels=LABELS,
                                                                               multi_label=True)
        if ADD_COORDS:
            X_train = np.concatenate((X_train, train_coords), axis=1)
        print("Fitting with min_samples_leaf={}".format(MIN_SAMPLES_LEAF))
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
        stats["train_times"].append(time() - train_start)
        # ---------------------------------------Evaluation on test set---------------------------------------
        test_preds = clf.predict(X_test)

        pred_masks = dsutils.preds_to_data_mask(test_preds, test_coords, LABELS)

        if ((it + 1) == 1) or ((it + 1) == NUM_ITERS):
            dsutils.save_pred_masks(pred_masks, DATA_DIR, SUBJ_ID, FEATURES_NAME + "_iter_" + str(it + 1))

        y_test_sub = y_test[:, 1:]
        test_preds_sub = test_preds[:, 1:]

        test_acc = sklearn.metrics.accuracy_score(y_test_sub, test_preds_sub)
        test_f1_macro = sklearn.metrics.f1_score(y_test_sub, test_preds_sub, average='macro')
        test_f1s = sklearn.metrics.f1_score(y_test_sub, test_preds_sub, average=None)

        print("Accuracy: {:.5f}, F1_macro: {:.5f}".format(test_acc, test_f1_macro))
        stats["iters"].append(it + 1)
        stats["train_voxels"].append((X_train.shape[0]))
        stats["scores"].append(test_f1_macro)
        # for c, f1 in enumerate(test_f1s):
        #     print("F1 for {}: {:.5f}".format(LABELS[c + 1], f1))

    np.savez(save_path, iters=stats["iters"],
             scores=stats["scores"],
             train_voxels=stats["train_voxels"],
             train_times=stats["train_times"])
print("Simulation took {:.5f} seconds".format(time() - script_start))
