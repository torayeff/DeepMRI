from os.path import join
import numpy as np
import nibabel as nib
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
import sys

sys.path.append("/home/agajan/DeepMRI")
from deepmri import dsutils, visutils  # noqa: E402

# settings
SUBJ_ID = "784565"
DATA_DIR = "/home/agajan/experiment_DiffusionMRI/tractseg_data/"
DATA_PATH = join(DATA_DIR, SUBJ_ID, "data.nii.gz")
MASK_PATH = join(DATA_DIR, SUBJ_ID, "nodif_brain_mask.nii.gz")
TRACT_MASKS_PTH = join(DATA_DIR, SUBJ_ID, "tract_masks", "tract_masks.nii.gz")
LABELS = ["Other", "CG", "CST", "FX", "CC"]

# load data
DATA = nib.load(DATA_PATH).get_data()
MASK = nib.load(MASK_PATH).get_data()
TRACT_MASKS = nib.load(TRACT_MASKS_PTH).get_data()
TRACT_MASKS = TRACT_MASKS[:, :, :, 1:]  # remove background class
SAVE_PATH = join(DATA_DIR, SUBJ_ID, "pca_stats_unnorm.npz")

ncs = list(range(1, 30))
stats = {
    "n_components": [],
    "scores": []
}

for nc in ncs:
    print("n_components = {}".format(nc).center(100, "-"))
    features, _ = dsutils.make_pca_volume(DATA, MASK, n_components=nc, normalize=False)
    train_slices = [('sagittal', 72), ('coronal', 87), ('axial', 72)]
    train_masks = dsutils.create_data_masks(TRACT_MASKS, train_slices, LABELS)
    X_train, y_train, train_coords = dsutils.create_dataset_from_data_mask(features,
                                                                           train_masks,
                                                                           labels=LABELS,
                                                                           multi_label=True)
    X_test, y_test, test_coords = dsutils.create_dataset_from_data_mask(features,
                                                                        TRACT_MASKS,
                                                                        labels=LABELS,
                                                                        multi_label=True)
    clf = RandomForestClassifier(n_estimators=100,
                                 bootstrap=True,
                                 oob_score=True,
                                 random_state=0,
                                 n_jobs=-1,
                                 max_features='auto',
                                 class_weight='balanced',
                                 max_depth=None,
                                 min_samples_leaf=8)
    clf.fit(X_train, y_train)
    test_preds = clf.predict(X_test)
    test_f1_macro = sklearn.metrics.f1_score(y_test[:, 1:], test_preds[:, 1:], average='macro')
    stats["n_components"].append(nc)
    stats["scores"].append(test_f1_macro)
    print(test_f1_macro)

np.savez(SAVE_PATH, n_components=stats["n_components"], scores=stats["scores"])
