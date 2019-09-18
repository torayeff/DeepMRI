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
FEATURES = np.zeros((145, 174, 145, 1))
LABELS = ["Other", "CG", "CST", "FX", "CC"]

# ---------------------------------------------Load Data----------------------------------------------

print("Loading Data".center(100, "-"))

TRACT_MASKS = nib.load(TRACT_MASKS_PTH).get_data()
TRACT_MASKS = TRACT_MASKS[:, :, :, 1:]  # remove background class

# ---------------------------------------------Train Set----------------------------------------------

print('Preparing the training set'.center(100, '-'))

train_slices = [('sagittal', 72), ('coronal', 87), ('axial', 72)]
train_masks = dsutils.create_data_masks(TRACT_MASKS, train_slices, LABELS)

X_train, y_train, train_coords = dsutils.create_dataset_from_data_mask(FEATURES,
                                                                       train_masks,
                                                                       labels=LABELS,
                                                                       multi_label=True)
print("X_train shape: {}, y_train shape: {}".format(X_train.shape, y_train.shape))

# dumb classifier
clf= {}
for i, crd in enumerate(train_coords):
    key = "{}_{}_{}".format(crd[0], crd[1], crd[2])
    clf[key] = y_train[i]
# ----------------------------------------------Test Set----------------------------------------------

print('Preparing the test set'.center(100, '-'))
test_slices = [('sagittal', 71), ('coronal', 86), ('axial', 71)]
test_masks = dsutils.create_data_masks(TRACT_MASKS, test_slices, LABELS)
X_test, y_test, test_coords = dsutils.create_dataset_from_data_mask(FEATURES,
                                                                    test_masks,
                                                                    labels=LABELS,
                                                                    multi_label=True)
print("X_test shape: {}, y_test shape: {}".format(X_test.shape, y_test.shape))

# ---------------------------------------Evaluation on test set---------------------------------------

print('Evaluating on the test set'.center(100, '-'))
test_preds = []
for i, crd in enumerate(test_coords):
    key = "{}_{}_{}".format(crd[0], crd[1], crd[2])
    if key in clf.keys():
        test_preds.append(clf[key])
        print(clf[key], "<=>", y_test[i])
    else:
        test_preds.append(np.array([1, 0, 0, 0, 0]))
test_preds = np.array(test_preds)

pred_masks = dsutils.preds_to_data_mask(test_preds, test_coords, LABELS)

y_test = y_test[:, 1:]
test_preds = test_preds[:, 1:]

test_acc = sklearn.metrics.accuracy_score(y_test, test_preds)
test_f1_macro = sklearn.metrics.f1_score(y_test, test_preds, average='macro')
test_f1s = sklearn.metrics.f1_score(y_test, test_preds, average=None)

print("Accuracy: ", test_acc)
for c, f1 in enumerate(test_f1s):
    print("F1 for {}: {:.5f}".format(LABELS[c+1], f1))
print("F1_macro: {:.5f}".format(test_f1_macro))
