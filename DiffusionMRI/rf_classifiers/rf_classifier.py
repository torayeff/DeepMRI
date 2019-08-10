import sys
from os.path import join
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics

sys.path.append('/home/agajan/DeepMRI')
from deepmri import dsutils  # noqa: E402

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
features_name = "raw_288_voxels"
min_samples_leaf = 9
import nibabel as nib
features = nib.load(join(data_dir, subj_id, 'data.nii.gz')).get_data()
# -----------------------------------------Prepare train set------------------------------------------
print('Prepare train set'.center(100, '-'))
train_slices = [('sagittal', 72), ('coronal', 87), ('axial', 72)]
train_masks = dsutils.create_data_masks(ml_masks, train_slices, labels)
X_train, y_train, train_coords = dsutils.create_dataset_from_data_mask(features,
                                                                       train_masks,
                                                                       labels=labels,
                                                                       multi_label=True)
# X_train = np.concatenate((X_train, train_coords), axis=1)
print("Trainset shape: ", X_train.shape, y_train.shape)
dsutils.label_stats_from_y(y_train, labels)
# ------------------------------------------Prepare test set------------------------------------------
print('Prepare test set'.center(100, '-'))
print("Test set is the whole brain volume.")
# test_masks = ml_masks
test_slices = [('sagittal', 71), ('coronal', 86), ('axial', 71)]
test_masks = dsutils.create_data_masks(ml_masks, test_slices, labels)
X_test, y_test, test_coords = dsutils.create_dataset_from_data_mask(features,
                                                                    test_masks,
                                                                    labels=labels,
                                                                    multi_label=True)
# X_test = np.concatenate((X_test, test_coords), axis=1)
print("Testset shape: ", X_test.shape, y_test.shape)
# --------------------------------------Random Forest Classifier--------------------------------------
print('Random Forest Classifier'.center(100, '-'))
clf = RandomForestClassifier(n_estimators=100,
                             bootstrap=True,
                             oob_score=True,
                             random_state=0,
                             n_jobs=-1,
                             max_features='auto',
                             class_weight='balanced',
                             max_depth=None,
                             min_samples_leaf=min_samples_leaf)
print("Fitting classiffier.")
clf.fit(X_train, y_train)

# --------------------------------------Evaluation on train set---------------------------------------
print('Evaluation on train set'.center(100, '-'))
train_preds = clf.predict(X_train)
train_acc = sklearn.metrics.accuracy_score(y_train, train_preds)
train_f1_macro = sklearn.metrics.f1_score(y_train, train_preds, average='macro')
train_f1s = sklearn.metrics.f1_score(y_train, train_preds, average=None)
print("Accuracy: {:.5f}, F1_macro: {:.5f}".format(train_acc, train_f1_macro))
for c, f1 in enumerate(train_f1s):
    print("F1 for {}: {:.5f}".format(labels[c], f1))
# ---------------------------------------Evaluation on test set---------------------------------------
print('Evaluation on test set'.center(100, '-'))
test_preds = clf.predict(X_test)

pred_masks = dsutils.preds_to_data_mask(test_preds, test_coords, labels)
dsutils.save_pred_masks(pred_masks, data_dir, subj_id, features_name)

y_test = y_test[:, 1:]
test_preds = test_preds[:, 1:]

test_acc = sklearn.metrics.accuracy_score(y_test, test_preds)
test_f1_macro = sklearn.metrics.f1_score(y_test, test_preds, average='macro')
test_f1s = sklearn.metrics.f1_score(y_test, test_preds, average=None)

print("Accuracy: {:.5f}, F1_macro: {:.5f}".format(test_acc, test_f1_macro))
for c, f1 in enumerate(test_f1s):
    print("F1 for {}: {:.5f}".format(labels[c + 1], f1))
