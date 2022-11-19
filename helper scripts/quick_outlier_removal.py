from data_utils.MastersDataset import MastersDataset
from pathlib import Path
import numpy as np
import pptk

file = Path(
    '/home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/data/PatrickData/Piazza')

ds = MastersDataset(None, file, sample_all_points=True)

id, cnt = np.unique(ds.grid_mask, return_counts=True)
rid = np.where(cnt > 10)
idxs = np.in1d(ds.grid_mask, rid)
v = pptk.viewer(ds.segment_points[0][:, :3], idxs==False, ds.segment_labels[0], np.logical_and(ds.segment_labels[0]==0,idxs==False))
v.set(selected=np.where(idxs == True))
np.save('Piazza', np.column_stack((ds.segment_points[0][idxs], ds.segment_labels[0][idxs])))
