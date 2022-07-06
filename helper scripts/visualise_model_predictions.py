import numpy as np
import torch
from pathlib import Path
import pptk

import Visualisation_utils
from data_utils.MastersDataset import MastersDataset
import models.pointnet2_sem_seg as Model
# import models.pointnet2_sem_seg_msg as Model

from tqdm import tqdm

log_path = Path(
    '/home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/log/masters/relative_coords')
data_path = Path(
    '/home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/data/PatrickData/Church/MastersFormat/hand_selected_reversed')

model_checkpoint_path = log_path / 'checkpoints/best_model.pth'

model_checkpoint = torch.load(model_checkpoint_path)
print(f"Reached best validation_IoU at epoch {model_checkpoint['epoch']}\n"
      f"IoU: {model_checkpoint['class_avg_iou']}")

# Check this in the logs on wandb
classifier = Model.get_model(2, points_vector_size=4)
classifier.load_state_dict(model_checkpoint['model_state_dict'])
classifier.cuda()

dataset = MastersDataset("validate", data_path, sample_all_points=True)

grid_data = dataset.__getitem__(0)
BATCH_SIZE = 16
available_batches = grid_data[0].shape[0]
num_batches = int(np.ceil(available_batches / BATCH_SIZE))

all_eval_points, all_eval_pred, all_eval_target, all_eval_probs = [], [], [], []

val_data_loader = torch.utils.data.DataLoader(dataset, batch_size=16,
                                              shuffle=True, num_workers=0, pin_memory=True,
                                              drop_last=False)
classifier.eval()
for i, (points, target_labels) in tqdm(enumerate(val_data_loader), total=len(val_data_loader), desc="Validation"):

    centers = points[:, :, :2].mean(axis=1)
    for center in centers:
        center

    if torch.is_tensor(points):
        points = points.data.numpy()
    points = torch.Tensor(points)
    if torch.is_tensor(target_labels):
        target_labels = target_labels.data.numpy()
    target_labels = torch.Tensor(target_labels)
    points, target_labels = points.float().cuda(), target_labels.long().cuda()
    points = points.transpose(2, 1)
    pred_logits, trans_feat, pred_choice = [], None, None

    pred_logits, trans_feat = classifier(points)

    pred_logits = pred_logits.contiguous().view(-1, 2)
    pred_choice = pred_logits.cpu().data.max(1)[1].numpy()
    batch_labels = target_labels.view(-1, 1)[:, 0].cpu().data.numpy()
    target_labels = target_labels.view(-1, 1)[:, 0]

    points = np.array(points.transpose(1, 2).cpu())
    preds = pred_choice.reshape(points.shape[0], -1)
    target = np.array(target_labels.cpu()).astype('int8').reshape(points.shape[0], -1)
    all_eval_points.append(points)
    all_eval_pred.append(preds)
    all_eval_target.append(target)
    all_eval_probs.append(pred_logits[:, 0].detach().exp().cpu().numpy().reshape(-1, dataset.num_points_in_block))
    continue

all_eval_points, all_eval_pred, all_eval_target, all_eval_probs = np.vstack(np.vstack(all_eval_points)), np.hstack(
    np.vstack(all_eval_pred)), np.hstack(np.vstack(all_eval_target)), np.hstack(np.vstack(all_eval_probs))
print("Showing intensity, predictions, target, difference, probability")
v = pptk.viewer(all_eval_points[:, :3], all_eval_points[:, 3], all_eval_pred, all_eval_target,
                all_eval_pred != all_eval_target)

total_seen_class, total_correct_class, total_iou_denominator_class = [0, 0], [0, 0], [0, 0]
for l in range(2):
    target_l = (all_eval_target == l)
    pred_l = (all_eval_pred == l)

    total_seen_class[l] += np.sum(target_l)  # How many times the label was available
    # How often the predicted label was correct in the batch
    total_correct_class[l] += np.sum(pred_l & target_l)
    # Total predictions + Class occurrences (Union prediction of class (right or wrong) and actual class occurrences.)
    total_iou_denominator_class[l] += np.sum((pred_l | target_l))

IoU = np.array(total_correct_class) / (np.array(total_iou_denominator_class,
                                                dtype=np.float64) + 1e-6)  # correct prediction/class occurrences + false prediction
mIoU = np.mean(IoU)
print(f"IoU keep: {IoU[0]}\n"
      f"IoU discard: {IoU[1]}\n"
      f"mIoU: {mIoU}")
pass
