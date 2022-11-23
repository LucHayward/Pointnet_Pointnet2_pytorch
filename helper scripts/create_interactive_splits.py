import numpy as np
import torch
# import pptk
import open3d as o3d
from pathlib import Path

import Visualisation_utils
from data_utils.MastersDataset import MastersDataset

path = Path("data/PatrickData/Church/MastersFormat/hand_selected_reversed")

dataset_t = MastersDataset("train", path)
dataset_v = MastersDataset("validate", path)

# vt = Visualisation_utils.pptk_full_dataset(dataset_t)
# vv = Visualisation_utils.pptk_full_dataset(dataset_v)

# selected = v.get("selected")
# output = np.hstack((dataset.segment_points[0][selected], dataset.segment_labels[0][selected,None]))
# np.save("train.npy", output)

# # Mask to select only those NOT selected
# mask = np.ones(dataset_v.segment_labels[0].size, dtype=bool)
# mask[selected] = False
#
# output_val = np.hstack((dataset_v.segment_points[0][mask], dataset_v.segment_labels[0][mask,None]))
# pptk.viewer(output_val[:,:3], output_val[:,-1] )
# np.save(Path("data/PatrickData/Church/MastersFormat/hand_selected_reversed_extra10%/validate.npy"), output_val)
#
# output_new = np.hstack((dataset_t.segment_points[0][selected], dataset_t.segment_labels[0][selected,None]))
# np.save(Path("data/PatrickData/Church/MastersFormat/hand_selected_reversed_extra10%_only_new/train.npy"), output_new)
# vt = Visualisation_utils.pptk_full_dataset(dataset_t)
# vv = Visualisation_utils.pptk_full_dataset(dataset_v)
