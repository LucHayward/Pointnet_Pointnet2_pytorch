import numpy as np
import torch
from pathlib import Path
import pptk

import Visualisation_utils
from data_utils.MastersDataset import MastersDataset
import models.pointnet2_sem_seg as Model

from tqdm import tqdm


# Get the path to the training run
log_path = Path("/home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/log/active_learning/AL: initial_test")

iteration = str(0)
# Show best training models predictions on the train and validation
with np.load(log_path/iteration/'train'/'train_predictions.npz') as train_pred_file:
    train_points = train_pred_file['points']
    train_preds = train_pred_file['preds']
    train_target = train_pred_file['target']
    train_diff = train_preds != train_target
with np.load(log_path/iteration/'train'/'val_predictions.npz') as val_pred_file:
    val_points = val_pred_file['points']
    val_preds = val_pred_file['preds']
    val_target = val_pred_file['target']
    val_diff = val_preds != val_target

vt = pptk.viewer(train_points[:,:3], train_points[:,3], train_preds, train_target, train_diff)
vv = pptk.viewer(val_points[:,:3], val_points[:,3], val_preds, val_target, val_diff)
