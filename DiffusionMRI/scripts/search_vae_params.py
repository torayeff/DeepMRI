import torch
import sys
from os.path import join
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
import time
import pickle

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, dsutils  # noqa: E402
from DiffusionMRI.ConvVAE import ConvVAE # noqa: E402

st = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
torch.backends.cudnn.benchmark = True  # set False whenever input size varies

# ------------------------------------------Settings--------------------------------------------------------------------
subj_id = '784565'
model_name = "Model15"
features_shape = (174, 145, 145, 22)
epochs = [200]
labels = ['Other', 'CG', 'CST', 'FX', 'CC']
mdps = [12, 15, 20, 25, 100, None]
msls = [1, 2, 4, 8, 16]
# mdps = [100]
# msls = [4]
epoch_best_params = {}

experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
data_dir = '/home/agajan/experiment_DiffusionMRI/tractseg_data/'
data_path = join(experiment_dir, 'tractseg_data', subj_id, 'training_slices/coronal')
masks_path = join(data_dir, subj_id, 'tract_masks/multi_label_mask.npz')

# ----------------------------------------------------------------------------------------------------------------------
ml_masks = np.load(masks_path)['data']
ml_masks = ml_masks[:, :, :, 1:]  # remove background class and other class

dataset = Datasets.OrientationDataset(data_path,
                                      scale=True,
                                      normalize=False,
                                      bg_zero=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=10)

model = ConvVAE(128, device)
model.to(device)
model.eval()
# ----------------------------------------------------------------------------------------------------------------------
for epoch in epochs:
    print("{} epoch: {}".format(model_name, epoch).center(100, '-'))
    model_path = "{}saved_models/{}_epoch_{}".format(experiment_dir, model_name, epoch)
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        features = torch.zeros(features_shape)

        for j, data in enumerate(dataloader):
            feature = data['data'].to(device)

            z_mean, z_log_var, encoded = model.encoder(feature)
            feature = model.decoder(encoded, return_feature=True)
            feature = feature.detach().cpu().squeeze().permute(1, 2, 0)

            idx = int(data['file_name'][0][:-4][-3:])
            features[idx] = feature

        features = features.numpy()

        # transpose features
        features = features.transpose(1, 0, 2, 3)

    print("Features from epoch={} were created".format(epoch))

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
    dims = test_coords.max(0) + 1
    idxs_to_remove = np.where(np.in1d(np.ravel_multi_index(test_coords.T, dims),
                                      np.ravel_multi_index(train_coords.T, dims)))[0]
    print(idxs_to_remove.shape)
    X_test = np.delete(X_test, idxs_to_remove, 0)
    y_test = np.delete(y_test, idxs_to_remove, 0)
    print("Testset shape after cleaning: ", X_test.shape, y_test.shape)

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
    print('Best F1: {}, Best max_depth: {}, Best min_samples_leaf:{}\n\n'.format(best_score, best_depth, best_leaf))
    epoch_best_params[epoch] = (best_score, best_depth, best_leaf)

# save
with open("epoch_best_params", "wb") as f:
    pickle.dump(epoch_best_params, f)

print('Runtime: {:.5f} seconds'.format(time.time() - st))
