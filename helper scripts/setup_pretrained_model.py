import numpy as np
import torch
import pptk
import open3d as o3d
from pathlib import Path

import models.pointnet2_sem_seg as MODEL

checkpoint = torch.load(
    "/home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/log/masters/best_model_pretrainedS3DIS.pth")

classifier = MODEL.get_model(13, points_vector_size=4)
classifier.load_state_dict(checkpoint['model_state_dict'])

import torch.nn as nn

classifier.conv2 = nn.Conv1d(128, 2, 1)
state = {
    'epoch': 0,
    'model_state_dict': classifier.state_dict(),
}

torch.save(state, Path(
    "log/active_learning/AL: testing/train/checkpoints/best_model.pth"))
