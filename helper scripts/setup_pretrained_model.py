import numpy as np
import torch
import pptk
import open3d as o3d
from pathlib import Path

import models.pointnet2_sem_seg as MODEL

# Load the pretrained model (13 classes)
checkpoint = torch.load("/home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/log/sem_seg/pretrained_pointnet2_sem_seg/checkpoints/best_model.pth")
# checkpoint = torch.load("/home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/log/masters/best_model_pretrainedS3DIS.pth")


# classifier = MODEL.get_model(13, points_vector_size=4)
classifier = MODEL.get_model(13)
classifier.load_state_dict(checkpoint['model_state_dict'])

import torch.nn as nn

# Create a new output layer with our class labels
classifier.conv2 = nn.Conv1d(128, 2, 1)
state = {
    'epoch': 0,
    'model_state_dict': classifier.state_dict(),
    'optimizer_state_dict': checkpoint['optimizer_state_dict']
}
savepath = Path("/home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/log/active_learning/5%_start_pretrained_all_layers_WD1e-2/0/train/checkpoints")
savepath.mkdir(parents=True, exist_ok=True)
torch.save(state, savepath/"best_model.pth")
