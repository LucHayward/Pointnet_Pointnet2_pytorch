import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes, points_vector_size=9, dropout_prob=0.5, sa1_groupall=False, sa2_groupall=False,
                 sa3_groupall=False, sa4_groupall=False, sa1_radius=0.1, sa2_radius=0.2, sa3_radius=0.4,
                 sa4_radius=0.8, sa1_npoint=1024, sa2_npoint=256, sa3_npoint=64, sa4_npoint=16):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(sa1_npoint, sa1_radius, 32, points_vector_size + 3, [32, 32, 64],
                                          sa1_groupall)
        self.sa2 = PointNetSetAbstraction(sa2_npoint, sa2_radius, 32, 64 + 3, [64, 64, 128], sa2_groupall)
        self.sa3 = PointNetSetAbstraction(sa3_npoint, sa3_radius, 32, 128 + 3, [128, 128, 256], sa3_groupall)
        self.sa4 = PointNetSetAbstraction(sa4_npoint, sa4_radius, 32, 256 + 3, [256, 256, 512], sa4_groupall)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])  # Not Random numbers, 256+512=sa4+sa3 output points
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])  # Gets us back to the original number of points
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, get_features=False, repeats=1):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]  # Just the XYZ values (TODO lets make these global?)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # Gets us back to the original number of points

        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = []
        for i in range(repeats):
            x.append(self.drop1(feat))
            x[i] = self.conv2(x[i])
            x[i] = F.log_softmax(x[i], dim=1)
            x[i] = x[i].permute(0, 2, 1)
        if repeats == 1:
            x = x[0]
        if get_features:
            return x, feat
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss


if __name__ == '__main__':
    import torch

    model = get_model(13)
    criterion = get_loss()
    xyz = torch.rand(6, 9, 2048)
    seg_pred, trans_feat = model(xyz)
    seg_pred = seg_pred.contiguous().view(-1, 13)
    pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
