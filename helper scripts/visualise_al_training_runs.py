import numpy as np
import torch
from pathlib import Path
import pptk

import Visualisation_utils
from data_utils.MastersDataset import MastersDataset
import models.pointnet2_sem_seg as Model

from tqdm import tqdm


def visualise_predicitons(merged=False):
    train_attr = [np.column_stack((train_preds, train_points[:, [3, 3]])),
                  np.column_stack((train_target, train_points[:, [3, 3]])),
                  np.column_stack((train_diff, train_points[:, [3, 3]]))]
    val_attr = [np.column_stack((val_preds, val_points[:, [3, 3]])),
                np.column_stack((val_target, val_points[:, [3, 3]])),
                np.column_stack((val_diff, val_points[:, [3, 3]]))]

    vt = pptk.viewer(train_points[:, :3], train_attr[0], train_attr[1], train_attr[2])
    vv = pptk.viewer(val_points[:, :3], val_attr[0], val_attr[1], val_attr[2])
    if merged:
        vm = pptk.viewer(np.vstack((train_points[:, :3], val_points[:, :3])), np.vstack((train_attr[0], val_attr[0])),
                     np.vstack((train_attr[1], val_attr[1])), np.vstack((train_attr[2], val_attr[2])))
        return vt, vv, vm
    else:
        return vt, vv


# Get the path to the training run
log_path = Path("/home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/log/active_learning/AL: initial_test")
for iteration in tqdm(range(11), desc="Al: iteration"):
    file_paths = list(Path(log_path / str(iteration) / 'train').glob('train*.npz'))
    file_paths.sort()
    if input("Visualise epoch checkpoints? (y/n)") == "y":
        for i in range(len(file_paths) - 1):
            print('Show epoch checkpoint models predictions on the train and validation')
            with np.load(log_path / str(iteration) / 'train' / f'train_predictions_epoch{i}.npz') as train_pred_file:
                train_points = train_pred_file['points']
                train_preds = train_pred_file['preds']
                train_target = train_pred_file['target']
                train_diff = train_preds != train_target
            with np.load(log_path / str(iteration) / 'train' / f'val_predictions_epoch{i}.npz') as val_pred_file:
                val_points = val_pred_file['points']
                val_preds = val_pred_file['preds']
                val_target = val_pred_file['target']
                val_diff = val_preds != val_target
            vv, vt = visualise_predicitons()
            if input("Enter to go next epoch") == 'n': break

    print('Show best training models predictions on the train and validation')
    with np.load(log_path / str(iteration) / 'train' / 'train_predictions.npz') as train_pred_file:
        train_points = train_pred_file['points']
        train_preds = train_pred_file['preds']
        train_target = train_pred_file['target']
        train_diff = train_preds != train_target
    with np.load(log_path / str(iteration) / 'train' / 'val_predictions.npz') as val_pred_file:
        val_points = val_pred_file['points']
        val_preds = val_pred_file['preds']
        val_target = val_pred_file['target']
        val_diff = val_preds != val_target

    vv, vt, vm = visualise_predicitons(True)

    # vt = pptk.viewer(train_points[:,:3], np.column_stack((train_preds,train_points[:,[3,3]])), np.column_stack((train_target,train_points[:,[3,3]])), np.column_stack((train_diff,train_points[:,[3,3]])))
    # vv = pptk.viewer(val_points[:,:3], np.column_stack((val_preds,val_points[:,[3,3]])), np.column_stack((val_target,val_points[:,[3,3]])), np.column_stack((val_diff,val_points[:,[3,3]])))
    input("Enter to go next iteration")
