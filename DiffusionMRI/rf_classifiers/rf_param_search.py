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
ml_masks = ml_masks[:, :, :, 1:]  # remove background class and other class

# -----------------------------------------Load Features------------------------------------------
# features = np.load(join(data_dir, subj_id, 'shore_features/shore_coefficients_radial_border_6.npz'))['data']
features = np.load(join(data_dir, subj_id, 'learned_features/Model10_features_epoch_900.npz'))['data']
# features = np.load(join(data_dir, subj_id, 'learned_features/SHORE_denoising_features_epoch_10000.npz'))['data']
# import nibabel as nib
# features = nib.load(join(data_dir, subj_id, 'data.nii.gz')).get_data()
print(features.shape)
# -----------------------------------------Prepare train set------------------------------------------
print('Prepare train set'.center(100, '-'))
train_slices = [('sagittal', 72), ('coronal', 87), ('axial', 72)]
train_masks = dsutils.create_data_masks(ml_masks, train_slices, labels)
X_train, y_train, train_coords = dsutils.create_dataset_from_data_mask(features,
                                                                       train_masks,
                                                                       multi_label=True)
print("Trainset shape: ", X_train.shape)

# ------------------------------------------Prepare test set------------------------------------------
print('Prepare test set'.center(100, '-'))
print("Test set is the whole brain volume.")
# test_masks = ml_masks
test_slices = [('sagittal', 71), ('coronal', 86), ('axial', 71)]
test_masks = dsutils.create_data_masks(ml_masks, test_slices, labels)
X_test, y_test, test_coords = dsutils.create_dataset_from_data_mask(features,
                                                                    test_masks,
                                                                    multi_label=True)
print("Testset shape: ", X_test.shape)
print("Removing train set from test set. Or do we want to include train set also in test set???")
dims = test_coords.max(0)+1
idxs_to_remove = np.where(np.in1d(np.ravel_multi_index(test_coords.T, dims),
                                  np.ravel_multi_index(train_coords.T, dims)))[0]
print(idxs_to_remove.shape)
X_test = np.delete(X_test, idxs_to_remove, 0)
y_test = np.delete(y_test, idxs_to_remove, 0)
print("Testset shape after cleaning: ", X_test.shape, y_test.shape)


mdps = [12, 15, 20, 25, 100, None]
msls = [1, 2, 4, 8, 16]

# mdps = [100]
# msls = [4]

train_scores = []
test_scores = []
best_score = 0
best_depth = None
best_leaf = None
for max_depth in mdps:
    for minleaf in msls:
        # --------------------------------------Random Forest Classifier--------------------------------------
        print('Random Forest Classifier'.center(100, '-'))
        clf = RandomForestClassifier(n_estimators=100,
                                     bootstrap=True,
                                     oob_score=True,
                                     random_state=0,
                                     n_jobs=-1,
                                     max_features='auto',
                                     class_weight='balanced',
                                     max_depth=max_depth,
                                     min_samples_leaf=minleaf)
        print("Fitting classiffier.")
        clf.fit(X_train, y_train)

        # ---------------------------------------Evaluation on test set---------------------------------------
        test_preds = clf.predict(X_test)
        print(test_preds.shape, y_test.shape)
        test_f1_macro = sklearn.metrics.f1_score(y_test[:, 1:], test_preds[:, 1:], average='macro')
        print('Test score: {}, Max depth: {}, Min leaf: {}'.format(test_f1_macro, max_depth, minleaf))
        if test_f1_macro > best_score:
            best_score = test_f1_macro
            best_depth = max_depth
            best_leaf = minleaf

print('Results'.center(100, '-'))
print('Best F1: {}, Best max_depth: {}, Best min_samples_leaf:{}'.format(best_score, best_depth, best_leaf))

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

test_acc = sklearn.metrics.accuracy_score(y_test, test_preds)
test_f1_macro = sklearn.metrics.f1_score(y_test, test_preds, average='macro')
test_f1s = sklearn.metrics.f1_score(y_test, test_preds, average=None)

print("Accuracy: {:.5f}, F1_macro: {:.5f}".format(test_acc, test_f1_macro))
for c, f1 in enumerate(test_f1s):
    print("F1 for {}: {:.5f}".format(labels[c+1], f1))
print('Runtime: {:.5f}'.format(time.time() - st))
