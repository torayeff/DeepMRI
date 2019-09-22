import torch
import sys
from os.path import join
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
import nibabel as nib
from time import time

sys.path.append('/home/agajan/DeepMRI')
from deepmri import Datasets, dsutils  # noqa: E402
from DiffusionMRI.models.ConvModel1 import Encoder, Decoder  # noqa: E402

# ------------------------------------------Settings--------------------------------------------------------------------
experiment_dir = '/home/agajan/experiment_DiffusionMRI/'
subj_id = '784565'
data_path = join(experiment_dir, 'tractseg_data', subj_id, "training_slices/coronal/")
LABELS = ["Other", "CG", "CST", "FX", "CC"]
DATA_DIR = "/home/agajan/experiment_DiffusionMRI/tractseg_data/"
TRACT_MASKS_PTH = join(DATA_DIR, subj_id, "tract_masks", "tract_masks.nii.gz")
TRACT_MASKS = nib.load(TRACT_MASKS_PTH).get_data()
TRACT_MASKS = TRACT_MASKS[:, :, :, 1:]  # remove background class


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device
deterministic = True  # reproducibility
seed = 0  # random seed for reproducibility
if deterministic:
    torch.manual_seed(seed)
torch.backends.cudnn.benchmark = (not deterministic)  # set False whenever input size varies
torch.backends.cudnn.deterministic = deterministic

batch_size = 1
start_epoch = 10
orient = "coronal"
# trainset = Datasets.VoxelDataset(data_path, file_name='data.nii.gz', normalize=False, scale=True)
trainset = Datasets.OrientationDataset(data_path, scale=True, normalize=False, bg_zero=True)
total_examples = len(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=6)
print("Total training examples: {}, Batch size: {}, Iters per epoch: {}".format(total_examples,
                                                                                batch_size,
                                                                                len(trainloader)))
stats = {
    "dims": [],
    "scores": []
}
best_score = 0
best_h = 10


for h in range(1, 101):
    hst = time()
    # model settings
    model_name = 'ConvModel1_prelu_h{}'.format(h)
    feature_shapes = (174, 145, 145, h)

    # encoder = Encoder(h=h)
    encoder = Encoder(input_size=(145, 145), h=h)
    decoder = Decoder(h=h)
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    encoder_path = "{}/saved_models/{}_encoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
    decoder_path = "{}/saved_models/{}_decoder_epoch_{}".format(experiment_dir, model_name, start_epoch)
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    print("Loaded pretrained weights starting from epoch {} for {}".format(start_epoch, model_name))

    # learned_features = np.zeros((145, 174, 145, h))
    # c = 0
    # with torch.no_grad():
    #     for data in trainloader:
    #         f = encoder(data['data'].to(device))
    #         for b in range(f.shape[0]):
    #             crd_0 = data['coord'][0][b].item()
    #             crd_1 = data['coord'][1][b].item()
    #             crd_2 = data['coord'][2][b].item()
    #             fvec = f[b].detach().cpu().squeeze().numpy().reshape(h)
    #             learned_features[crd_0, crd_1, crd_2] = fvec
    #         c += 1
    #         print(c, end=" ")
    with torch.no_grad():
        orient_features = torch.zeros(feature_shapes)
        for j, data in enumerate(trainloader):
            x = data['data'].to(device)
            feature = encoder(x)
            orient_feature = feature.detach().cpu().squeeze(0).permute(1, 2, 0)

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
        # ---------------------------------------------Train Set----------------------------------------------
        FEATURES = orient_features
        train_slices = [('sagittal', 72), ('coronal', 87), ('axial', 72)]
        train_masks = dsutils.create_data_masks(TRACT_MASKS, train_slices, LABELS, verbose=False)
        X_train, y_train, train_coords = dsutils.create_dataset_from_data_mask(FEATURES,
                                                                               train_masks,
                                                                               labels=LABELS,
                                                                               multi_label=True)

        # --------------------------------------Random Forest Classifier--------------------------------------
        clf = RandomForestClassifier(n_estimators=100,
                                     bootstrap=True,
                                     oob_score=True,
                                     random_state=0,
                                     n_jobs=-1,
                                     max_features='auto',
                                     class_weight='balanced',
                                     max_depth=None,
                                     min_samples_leaf=8)
        clf.fit(X_train, y_train)

        # ----------------------------------------------Test Set----------------------------------------------
        test_masks = TRACT_MASKS.copy()
        # test_slices = [('sagittal', 71), ('coronal', 86), ('axial', 71)]
        # test_masks = dsutils.create_data_masks(TRACT_MASKS, test_slices, LABELS, verbose=False)
        X_test, y_test, test_coords = dsutils.create_dataset_from_data_mask(FEATURES,
                                                                            test_masks,
                                                                            labels=LABELS,
                                                                            multi_label=True)
        # ---------------------------------------Evaluation on test set---------------------------------------
        test_preds = clf.predict(X_test)

        pred_masks = dsutils.preds_to_data_mask(test_preds, test_coords, LABELS)

        y_test = y_test[:, 1:]
        test_preds = test_preds[:, 1:]

        test_acc = sklearn.metrics.accuracy_score(y_test, test_preds)
        test_f1_macro = sklearn.metrics.f1_score(y_test, test_preds, average='macro')

        print("h: {}, F1: {:.5f}".format(h, test_f1_macro))
        stats["dims"].append(h)
        stats["scores"].append(test_f1_macro)

        if test_f1_macro > best_score:
            best_h = h
            best_score = test_f1_macro

        print("Done in {:.3f} secs.".format(time() - hst))
print("Best score: {:.5f}, best h: {}".format(best_score, best_h))
np.savez("ConvModel1_prelu" + "_h_search_fullbrain.npz", dims=stats["dims"], scores=stats["scores"])
