import torch
import os
import sys
from os.path import join
import numpy as np
import nibabel as nib
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, dsutils  # noqa: E402
from DiffusionMRI.models.Model10 import Encoder  # noqa: E402  # noqa: E402

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
torch.backends.cudnn.benchmark = True  # set False whenever input size varies

experiment_dir = '/home/agajan/experiment_DiffusionMRI/'

subj_id = '784565'
orients = ['coronal']
model_name = "Model10_new"
feature_shapes = [(174, 145, 145, 22)]
noise_prob = None
MIN_SAMPLES_LEAF = 3
LABELS = ["Other", "CG", "CST", "FX", "CC"]
FULL_BRAIN = True
ADD_COORDS = False
DATA_DIR = "/home/agajan/experiment_DiffusionMRI/tractseg_data/"
TRACT_MASKS_PTH = join(DATA_DIR, subj_id, "tract_masks", "tract_masks.nii.gz")

encoder = Encoder(input_size=(145, 145))
encoder.to(device)
encoder.eval()

epochs = list(range(10, 1001, 10))
for epoch in epochs:

    for i, orient in enumerate(orients):
        data_path = os.path.join(experiment_dir, 'tractseg_data', subj_id, 'training_slices', orient)

        dataset = Datasets.OrientationDataset(data_path,
                                              scale=True,
                                              normalize=False,
                                              bg_zero=True,
                                              noise_prob=noise_prob,
                                              alpha=1)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)

        encoder_path = "{}saved_models/{}_encoder_epoch_{}".format(experiment_dir, model_name, epoch)
        encoder.load_state_dict(torch.load(encoder_path))

        with torch.no_grad():
            orient_features = torch.zeros(feature_shapes[i])
            for j, data in enumerate(dataloader):
                x = data['data'].to(device)
                feature = encoder(x)
                orient_feature = feature.detach().cpu().squeeze().permute(1, 2, 0)

                idx = int(data['file_name'][0][:-4][-3:])
                orient_features[idx] = orient_feature

            orient_features = orient_features.numpy()
            # transpose features
            if orient == 'coronal':
                orient_features = orient_features.transpose(1, 0, 2, 3)
            if orient == 'axial':
                orient_features = orient_features.transpose(1, 2, 0, 3)

            TRACT_MASKS = nib.load(TRACT_MASKS_PTH).get_data()
            TRACT_MASKS = TRACT_MASKS[:, :, :, 1:]  # remove background class

            FEATURES = orient_features

            # ---------------------------------------------Train Set----------------------------------------------

            train_slices = [('sagittal', 72), ('coronal', 87), ('axial', 72)]
            train_masks = dsutils.create_data_masks(TRACT_MASKS, train_slices, LABELS, verbose=False)
            X_train, y_train, train_coords = dsutils.create_dataset_from_data_mask(FEATURES,
                                                                                   train_masks,
                                                                                   labels=LABELS,
                                                                                   multi_label=True)
            if ADD_COORDS:
                X_train = np.concatenate((X_train, train_coords), axis=1)

            # --------------------------------------Random Forest Classifier--------------------------------------
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

            # ----------------------------------------------Test Set----------------------------------------------
            if FULL_BRAIN:
                test_masks = TRACT_MASKS.copy()
            else:
                test_slices = [('sagittal', 71), ('coronal', 86), ('axial', 71)]
                test_masks = dsutils.create_data_masks(TRACT_MASKS, test_slices, LABELS, verbose=False)
            X_test, y_test, test_coords = dsutils.create_dataset_from_data_mask(FEATURES,
                                                                                test_masks,
                                                                                labels=LABELS,
                                                                                multi_label=True)
            if ADD_COORDS:
                X_test = np.concatenate((X_test, test_coords), axis=1)
            # ---------------------------------------Evaluation on test set---------------------------------------
            test_preds = clf.predict(X_test)

            pred_masks = dsutils.preds_to_data_mask(test_preds, test_coords, LABELS)

            y_test = y_test[:, 1:]
            test_preds = test_preds[:, 1:]

            test_acc = sklearn.metrics.accuracy_score(y_test, test_preds)
            test_f1_macro = sklearn.metrics.f1_score(y_test, test_preds, average='macro')

            print("Epoch: {}, F1: {:.5f}".format(epoch, test_f1_macro))
