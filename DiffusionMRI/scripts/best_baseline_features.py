import sys
import os
import numpy as np
import nibabel as nib
import time
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics

sys.path.append('/home/agajan/DeepMRI')
from deepmri import dsutils  # noqa: E402

# load data
data_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/'
subj_id = '784565'
dmri_path = os.path.join(data_dir, subj_id, 'data.nii.gz')
dmri = nib.load(dmri_path)
print("dMRI shape: ", dmri.shape)
dmri_data = dmri.get_fdata()

# load masks
masks_path = os.path.join(data_dir, subj_id, 'tract_masks')
ml_masks = np.load(os.path.join(masks_path, 'multi_label_mask.npz'))['data']
ml_masks = ml_masks[:, :, :, 1:]  # remove background class
print("Mask shape: ", ml_masks.shape)

labels = ['Other', 'CG', 'CST', 'FX', 'CC']

training_slices = [('sagittal', 72), ('coronal', 87), ('axial', 72)]

# make training data
train_masks = dsutils.create_data_masks(ml_masks, training_slices, labels)
X_train, y_train, train_coords = dsutils.create_dataset_from_data_mask(dmri_data,
                                                                       train_masks,
                                                                       labels=labels,
                                                                       multi_label=True)

# make test data
test_masks = ml_masks
X_test, y_test, test_coords = dsutils.create_dataset_from_data_mask(dmri_data,
                                                                    test_masks,
                                                                    labels=labels,
                                                                    multi_label=True)
X_train = np.hstack((train_coords, X_train))
X_test = np.hstack((test_coords, X_test))
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)

num_features = list(range(1, 289))
best_f1 = 0
best_n = None
test_f1_scores = []

for nf in num_features:
    st = time.time()

    clf = RandomForestClassifier(n_estimators=100,
                                 bootstrap=True,
                                 oob_score=True,
                                 random_state=0,
                                 n_jobs=-1,
                                 max_features='auto')

    clf.fit(X_train[:, :nf], y_train)

    test_preds = clf.predict(X_test[:, :nf])
    test_f1_macro = sklearn.metrics.f1_score(y_test, test_preds, average='macro')
    test_f1_scores.append(test_f1_macro)
    print("n_features: {}, f1_macro: {:.5f}, time: {:.5f}".format(nf, test_f1_macro, time.time() - st))

    if test_f1_macro > best_f1:
        best_f1 = test_f1_macro
        best_n = nf

np.savez('baseline_features.npz',
         num_features=num_features,
         test_f1_scores=test_f1_scores,
         best_f1=best_f1,
         best_n=best_n)

print("Best F1={:.5f} at n={}".format(best_f1, best_n))
