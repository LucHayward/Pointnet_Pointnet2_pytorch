import numpy as np
import torch
import pptk
import open3d as o3d
from pathlib import Path

import Visualisation_utils
from data_utils.MastersDataset import MastersDataset

path = Path("data/PatrickData/Church/MastersFormat/hand_selected_50%")

dataset = MastersDataset(None, path, sample_all_points=True)

v = Visualisation_utils.pptk_full_dataset(dataset, include_grid_mask=True)

