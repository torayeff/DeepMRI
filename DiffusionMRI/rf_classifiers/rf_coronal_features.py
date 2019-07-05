import sys
from os.path import join
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics

sys.path.append('/home/agajan/DeepMRI')
from deepmri import ds_utils  # noqa: E402

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

# load learned features
feature_name = 'Conv2dAECoronalStrided_features_epoch_50.npz'
features_path = join(data_dir, subj_id, 'learned_features', feature_name)
learned_features = np.load(features_path)['data']

# -----------------------------------------Prepare train set------------------------------------------
print('Prepare train set'.center(100, '-'))
train_slices = [('sagittal', 72), ('coronal', 87), ('axial', 72)]
train_masks = ds_utils.create_data_masks(ml_masks, train_slices, labels)
X_train, y_train, train_coords = ds_utils.create_dataset_from_data_mask(learned_features,
                                                                        train_masks,
                                                                        labels=labels,
                                                                        multi_label=True)
X_train = np.hstack((train_coords, X_train))
print("Trainset shape: ", X_train.shape)

# ------------------------------------------Prepare test set------------------------------------------
print('Prepare test set'.center(100, '-'))
print("Test set is the whole brain volume.")
test_masks = ml_masks
X_test, y_test, test_coords = ds_utils.create_dataset_from_data_mask(learned_features,
                                                                     test_masks,
                                                                     labels=labels,
                                                                     multi_label=True)
X_test = np.hstack((test_coords, X_test))
print("Testset shape: ", X_test.shape)

# --------------------------------------Random Forest Classifier--------------------------------------
print('Random Forest Classifier'.center(100, '-'))
clf = RandomForestClassifier(n_estimators=100,
                             bootstrap=True,
                             oob_score=True,
                             random_state=0,
                             n_jobs=-1,
                             max_features='auto',
                             class_weight='balanced',
                             max_depth=100,
                             min_samples_leaf=8)
print("Fitting classiffier.")
clf.fit(X_train, y_train)

# --------------------------------------Evaluation on train set---------------------------------------
print('Evaluation on train set'.center(100, '-'))
train_preds = clf.predict(X_train)
train_acc = sklearn.metrics.accuracy_score(y_train, train_preds)
train_f1_macro = sklearn.metrics.f1_score(y_train, train_preds, average='macro')
train_f1s = sklearn.metrics.f1_score(y_train, train_preds, average=None)
print("OOB Score: {:.5f}".format(clf.oob_score_))
print("Accuracy: {:.5f}, F1_macro: {:.5f}".format(train_acc, train_f1_macro))
for c, f1 in enumerate(train_f1s):
    print("F1 for {}: {:.5f}".format(labels[c], f1))

# ---------------------------------------Evaluation on test set---------------------------------------
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

