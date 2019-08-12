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
FEATURES_NAME = "MODEL6"
FEATURES_FILE = "shore_features/shore_coefficients_radial_border_4.npz"
FEATURES_FILE = "learned_features/Model6_features_epoch_200.npz"
ADD_COORDS = False
FEATURES_PATH = join(DATA_DIR, SUBJ_ID, FEATURES_FILE)
MIN_SAMPLES_LEAF = 10
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

# --------------------------------------Random Forest Classifier--------------------------------------

print('Random Forest Classifier'.center(100, '-'))
print("Fitting withh min_samples_leaf={}".format(MIN_SAMPLES_LEAF))

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

# --------------------------------------Evaluation on train set---------------------------------------

print('Evaluating on the train set'.center(100, '-'))
train_preds = clf.predict(X_train)
train_acc = sklearn.metrics.accuracy_score(y_train, train_preds)
train_f1_macro = sklearn.metrics.f1_score(y_train, train_preds, average='macro')
train_f1s = sklearn.metrics.f1_score(y_train, train_preds, average=None)
print("Accuracy: {:.5f}, F1_macro: {:.5f}".format(train_acc, train_f1_macro))
for c, f1 in enumerate(train_f1s):
    print("F1 for {}: {:.5f}".format(LABELS[c], f1))

# ----------------------------------------------Test Set----------------------------------------------

print('Preparing the test set'.center(100, '-'))
test_masks = TRACT_MASKS.copy()
# test_slices = [('sagittal', 71), ('coronal', 86), ('axial', 71)]
# test_masks = dsutils.create_data_masks(TRACT_MASKS, test_slices, LABELS)
X_test, y_test, test_coords = dsutils.create_dataset_from_data_mask(FEATURES,
                                                                    test_masks,
                                                                    labels=LABELS,
                                                                    multi_label=True)
if ADD_COORDS:
    X_test = np.concatenate((X_test, test_coords), axis=1)
print("X_test shape: {}, y_test shape: {}".format(X_test.shape, y_test.shape))

# ---------------------------------------Evaluation on test set---------------------------------------

print('Evaluating on the test set'.center(100, '-'))
test_preds = clf.predict(X_test)

pred_masks = dsutils.preds_to_data_mask(test_preds, test_coords, LABELS)
dsutils.save_pred_masks(pred_masks, DATA_DIR, SUBJ_ID, FEATURES_NAME)

y_test = y_test[:, 1:]
test_preds = test_preds[:, 1:]

test_acc = sklearn.metrics.accuracy_score(y_test, test_preds)
test_f1_macro = sklearn.metrics.f1_score(y_test, test_preds, average='macro')
test_f1s = sklearn.metrics.f1_score(y_test, test_preds, average=None)

print("Accuracy: {:.5f}, F1_macro: {:.5f}".format(test_acc, test_f1_macro))
for c, f1 in enumerate(test_f1s):
    print("F1 for {}: {:.5f}".format(LABELS[c + 1], f1))
