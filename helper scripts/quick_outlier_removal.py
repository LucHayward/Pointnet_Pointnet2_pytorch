from data_utils.MastersDataset import MastersDataset
from pathlib import Path
import numpy as np
import pptk

file = Path(
    '/home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/data/PatrickData/Church/MastersFormat/church_registered.npy')

ds = MastersDataset(None, file, sample_all_points=True)

id, cnt = np.unique(ds.grid_mask, return_counts=True)
rid = np.where(cnt > 100)
idxs = np.in1d(ds.grid_mask, rid)
v = pptk.viewer(ds.segment_points[0][:, :3])
v.set(selected=np.where(idxs == True))
np.save('file', np.column_stack((ds.segment_points[0][idxs], ds.segment_labels[0][idxs])))
