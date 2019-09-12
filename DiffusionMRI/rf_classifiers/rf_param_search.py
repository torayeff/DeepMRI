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
FEATURES_NAME = "EXP"
FEATURES_FILE = "data.nii.gz"
FULL_BRAIN = True
ADD_COORDS = False
FEATURES_PATH = join(DATA_DIR, SUBJ_ID, FEATURES_FILE)
LABELS = ["Other", "CG", "CST", "FX", "CC"]

# ---------------------------------------------Load Data----------------------------------------------

print("Loading Data".center(100, "-"))

TRACT_MASKS = nib.load(TRACT_MASKS_PTH).get_data()
TRACT_MASKS = TRACT_MASKS[:, :, :, 1:]  # remove background class

if FEATURES_PATH.endswith(".npz"):
    FEATURES = np.load(FEATURES_PATH)["data"]
else:
    FEATURES = nib.load(FEATURES_PATH).get_data()

print("FEATURES Name: {}, shape: {}".format(FEATURES_NAME, FEATURES.shape))

# ---------------------------------------------Train Set----------------------------------------------

print('Preparing the training set'.center(100, '-'))

train_slices = [('sagittal', 72), ('coronal', 87), ('axial', 72)]
train_masks = dsutils.create_data_masks(TRACT_MASKS, train_slices, LABELS)

X_train, y_train, train_coords = dsutils.create_dataset_from_data_mask(FEATURES,
                                                                       train_masks,
                                                                       labels=LABELS,
                                                                       multi_label=True)
if ADD_COORDS:
    X_train = np.concatenate((X_train, train_coords), axis=1)
print("X_train shape: {}, y_train shape: {}".format(X_train.shape, y_train.shape))

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

# --------------------------------------Random Forest Classifier--------------------------------------
print('Random Forest Classifier'.center(100, '-'))

msls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

train_scores = []
test_scores = []
best_score = 0
best_min_samples_leaf = None

for min_samples_leaf in msls:
    clf = RandomForestClassifier(n_estimators=100,
                                 bootstrap=True,
                                 oob_score=True,
                                 random_state=0,
                                 n_jobs=-1,
                                 max_features='auto',
                                 class_weight='balanced',
                                 max_depth=None,
                                 min_samples_leaf=min_samples_leaf)
    clf.fit(X_train, y_train)

    # ---------------------------------------Evaluation on test set---------------------------------------
    test_preds = clf.predict(X_test)
    test_f1_macro = sklearn.metrics.f1_score(y_test[:, 1:], test_preds[:, 1:], average='macro')
    print("Test score: {}, min_samples_leaf = {}".format(test_f1_macro, min_samples_leaf))
    if test_f1_macro > best_score:
        best_score = test_f1_macro
        best_depth = None
        best_leaf = min_samples_leaf

print('Results'.center(100, '-'))
print('Best F1: {}, Best min_samples_leaf={}'.format(best_score, best_leaf))

clf = RandomForestClassifier(n_estimators=100,
                             bootstrap=True,
                             oob_score=True,
                             random_state=0,
                             n_jobs=-1,
                             max_features='auto',
                             class_weight='balanced',
                             max_depth=None,
                             min_samples_leaf=best_leaf)
print("Fitting classiffier.")
clf.fit(X_train, y_train)
print('Evaluation on test set'.center(100, '-'))
test_preds = clf.predict(X_test)

y_test = y_test[:, 1:]
test_preds = test_preds[:, 1:]

test_f1_macro = sklearn.metrics.f1_score(y_test, test_preds, average='macro')
test_f1s = sklearn.metrics.f1_score(y_test, test_preds, average=None)

for c, f1 in enumerate(test_f1s):
    print("F1 for {}: {:.5f}".format(LABELS[c+1], f1))
print("F1_macro: {:.5f}".format(test_f1_macro))
