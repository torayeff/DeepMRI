import sys
from os.path import join
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
import time

sys.path.append('/home/agajan/DeepMRI')
from deepmri import dsutils  # noqa: E402
st = time.time()
# ------------------------------------Setting up and loading data-------------------------------------
print('Setting up and loading data'.center(100, '-'))
data_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/'
subj_id = '784565'
print('Experiment for subject id={}.'.format(subj_id))
labels = ['Other', 'CG', 'CST', 'FX', 'CC']

# load masks
masks_path = join(data_dir, subj_id, 'tract_masks')
ml_masks = np.load(join(masks_path, 'multi_label_mask.npz'))['data']
ml_masks = ml_masks[:, :, :, 1:]  # remove background class

# -----------------------------------------Load Features------------------------------------------
features = np.load(join(data_dir, subj_id, 'learned_features/final/Model2_features_epoch_100.npz'))['data']
# features = np.load(join(data_dir, subj_id, 'shore_features/shore_coefficients_radial_border_4.npz'))['data']
# import nibabel as nib
# features = nib.load(join(data_dir, subj_id, 'data.nii.gz')).get_data()
# features = np.load(join(data_dir, subj_id, 'avg_raw_nh9.npz'))['data']
# print(features1.shape, features2.shape)
# features = np.concatenate((features1, features2), axis=3)
# print(features.shape)
# -----------------------------------------Prepare train set------------------------------------------
print('Prepare train set'.center(100, '-'))
train_slices = [('sagittal', 72), ('coronal', 87), ('axial', 72)]
train_masks = dsutils.create_data_masks(ml_masks, train_slices, labels)
X_train, y_train, train_coords = dsutils.create_dataset_from_data_mask(features,
                                                                       train_masks,
                                                                       multi_label=True)
# X_train = train_coords
X_train = np.concatenate((X_train, train_coords), axis=1)
print(X_train.shape, train_coords.shape)
print("Trainset shape: ", X_train.shape)
dsutils.label_stats_from_y(y_train, labels)
# ------------------------------------------Prepare test set------------------------------------------
print('Prepare test set'.center(100, '-'))
print("Test set is the whole brain volume.")
test_masks = ml_masks
# test_slices = [('sagittal', 71), ('coronal', 86), ('axial', 71)]
# test_masks = dsutils.create_data_masks(ml_masks, test_slices, labels)
X_test, y_test, test_coords = dsutils.create_dataset_from_data_mask(features,
                                                                    test_masks,
                                                                    multi_label=True)
# X_test = test_coords
X_test = np.concatenate((X_test, test_coords), axis=1)
print("Testset shape: ", X_test.shape)
dsutils.label_stats_from_y(y_test, labels)
# --------------------------------------Random Forest Classifier--------------------------------------
print('Random Forest Classifier'.center(100, '-'))
mdps = [None]
msls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# msls = [11, 13, 14, 20]
# msls = [9]

train_scores = []
test_scores = []
best_score = 0
best_depth = None
best_leaf = None
for minleaf in msls:
    clf = RandomForestClassifier(n_estimators=100,
                                 bootstrap=True,
                                 oob_score=True,
                                 random_state=0,
                                 n_jobs=-1,
                                 max_features='auto',
                                 class_weight='balanced',
                                 max_depth=None,
                                 min_samples_leaf=minleaf)
    print("Fitting classiffier.")
    clf.fit(X_train, y_train)

    # ---------------------------------------Evaluation on test set---------------------------------------
    test_preds = clf.predict(X_test)
    test_f1_macro = sklearn.metrics.f1_score(y_test[:, 1:], test_preds[:, 1:], average='macro')
    print('Test score: {}, min_samples_leaf = {}'.format(test_f1_macro, minleaf))
    if test_f1_macro > best_score:
        best_score = test_f1_macro
        best_depth = None
        best_leaf = minleaf

print('Results'.center(100, '-'))
print('Best F1: {}, Best max_depth: {}, Best min_samples_leaf={}'.format(best_score, best_depth, best_leaf))

clf = RandomForestClassifier(n_estimators=100,
                             bootstrap=True,
                             oob_score=True,
                             random_state=0,
                             n_jobs=-1,
                             max_features='auto',
                             class_weight='balanced',
                             max_depth=best_depth,
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
    print("F1 for {}: {:.5f}".format(labels[c+1], f1))
print("F1_macro: {:.5f}".format(test_f1_macro))
print('Runtime: {:.5f}'.format(time.time() - st))
