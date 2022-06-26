import numpy as np
import torch
from pathlib import Path
import pptk

import Visualisation_utils
from data_utils.MastersDataset import MastersDataset
import models.pointnet2_sem_seg as Model

from tqdm import tqdm

log_path = Path(
    '/home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/log/masters/hand_selected_reversed_start_pretrained_all_layers')
model_checkpoint_path = log_path / 'checkpoints/best_model.pth'

model_checkpoint = torch.load(model_checkpoint_path)
print(f"Reached best validation_IoU at epoch {model_checkpoint['epoch']}\n"
      f"IoU: {model_checkpoint['class_avg_iou']}")

# Check this in the logs on wandb
classifier = Model.get_model(2, points_vector_size=4)
classifier.load_state_dict(model_checkpoint['model_state_dict'])
classifier.cuda()

dataset = MastersDataset("validate",
                         Path('/home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/data/PatrickData/Church/MastersFormat/hand_selected_reversed'),
                         sample_all_points=False)

grid_data = dataset.__getitem__(0)
BATCH_SIZE = 16
available_batches = grid_data[0].shape[0]
num_batches = int(np.ceil(available_batches / BATCH_SIZE))

all_eval_points, all_eval_pred, all_eval_target = [],[],[]

# for batch in tqdm(range(num_batches), desc="Validation"):
#     points, target_labels = grid_data[0][batch * BATCH_SIZE:batch * BATCH_SIZE + BATCH_SIZE], \
#                             grid_data[1][batch * BATCH_SIZE:batch * BATCH_SIZE + BATCH_SIZE]
val_data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                          shuffle=False, num_workers=0, pin_memory=True,
                                                          drop_last=(True if len(dataset) // BATCH_SIZE else False))
for i, (points, target_labels) in tqdm(enumerate(val_data_loader), total=len(val_data_loader),
                                                       desc="Validation"):

    if torch.is_tensor(points):
        points = points.data.numpy()
    points = torch.Tensor(points)
    if torch.is_tensor(target_labels):
        target_labels = target_labels.data.numpy()
    target_labels = torch.Tensor(target_labels)
    points, target_labels = points.float().cuda(), target_labels.long().cuda()
    points = points.transpose(2, 1)
    pred_labels, trans_feat, pred_choice = [], None, None

    # Run MC Dropout to get T sets of predictions.
    pred_labels, trans_feat = classifier(points)


    pred_labels = pred_labels.contiguous().view(-1, 2)
    pred_choice = pred_labels.cpu().data.max(1)[1].numpy()
    batch_labels = target_labels.view(-1, 1)[:, 0].cpu().data.numpy()
    target_labels = target_labels.view(-1, 1)[:, 0]

    points = np.array(points.transpose(1, 2).cpu())
    preds = pred_choice.reshape(points.shape[0], -1)
    target = np.array(target_labels.cpu()).astype('int8').reshape(points.shape[0], -1)
    all_eval_points.append(points)
    all_eval_pred.append(preds)
    all_eval_target.append(target)


all_eval_points, all_eval_pred, all_eval_target = np.vstack(np.vstack(all_eval_points)), np.hstack(np.vstack(all_eval_pred)), np.hstack(np.vstack(all_eval_target))
print("Showing intensity, predictions, target, difference")
v = pptk.viewer(all_eval_points[:,:3], all_eval_points[:,3], all_eval_pred, all_eval_target, all_eval_pred!= all_eval_target)


total_seen_class, total_correct_class, total_iou_denominator_class = [0, 0], [0, 0], [0, 0]
for l in range(2):
    target_l = (all_eval_target == l)
    pred_l = (all_eval_pred == l)

    total_seen_class[l] += np.sum(target_l)  # How many times the label was available
    # How often the predicted label was correct in the batch
    total_correct_class[l] += np.sum(pred_l & target_l)
    # Total predictions + Class occurrences (Union prediction of class (right or wrong) and actual class occurrences.)
    total_iou_denominator_class[l] += np.sum((pred_l | target_l))

mIoU = np.mean(
    np.array(total_correct_class) / (np.array(total_iou_denominator_class,
                                              dtype=np.float64) + 1e-6))  # correct prediction/class occurrences + false prediction

print(f"mIoU: {mIoU}")