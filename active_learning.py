import numpy as np
import torch
import pptk
import open3d as o3d
from pathlib import Path

import Visualisation_utils
from data_utils.MastersDataset import MastersDataset

TEMP_DIR = Path("temp/")


def save_split_dataset(dataset, selected_points, iteration: int):
    """
    Splits and saves the dataset out to the temp directory
    TODO change to the log dir
    :param dataset: dataset being selected from
    :param selected_points: selected train/labelling points
    :param iteration: which iteration of the Active Learning process
    :return:
    """
    train_points = dataset.segment_points[0][selected_points]
    train_labels = dataset.segment_labels[0][selected_points]
    np.save(TEMP_DIR / f"train{iteration}.npy", np.hstack((train_points, train_labels[:, None])))

    val_points = Visualisation_utils.numpy_inverse_index(dataset.segment_points[0], selected_points)
    val_labels = Visualisation_utils.numpy_inverse_index(dataset.segment_labels[0], selected_points)
    np.save(TEMP_DIR / f"validate{iteration}.npy", np.hstack((train_points, train_labels[:, None])))


def select_new_points_to_label(dataset, viewer):
    completed_selection = False
    num_grid_cells = len(dataset.grid_cell_to_segment)
    selected_label_idxs = None
    while not completed_selection:
        print(f"Select 5% of the cells ({num_grid_cells * .05:.0f}/{num_grid_cells}][) for labelling")
        input("Waiting for selection...(enter)")
        selected = viewer.get('selected')
        selected_cells = np.unique(dataset.grid_mask[selected])
        selected_labelled_idxs = np.where(np.in1d(dataset.grid_mask, selected_cells))[0]
        viewer.set(selected=selected_labelled_idxs)
        print(f"Selected {len(selected_labelled_idxs)} points from {len(selected_cells)} cells "
              f"({len(selected_labelled_idxs) / len(dataset.grid_mask) * 100:.2f}% of points, "
              f"{len(selected_cells) / num_grid_cells * 100:.2f}% of area)")
        completed_selection = input("Happy with selection? To adjust enter N, otherwise enter Y").upper() == "Y"

    return selected_labelled_idxs

def main():
    TEMP_DIR.mkdir(exist_ok=True)
    # get full pcd
    initial_dataset = MastersDataset(None, Path("data/PatrickData/Church/MastersFormat"), sample_all_points=True)
    v_init = Visualisation_utils.pptk_full_dataset(initial_dataset, include_grid_mask=True, include_intensity=True)
    v_init.color_map(
        Visualisation_utils.turbo_colormap_data[::16] * 16)  # Repeats colours distinguishing adjacent cells.
    v_init.color_map("summer")  # Best for intensity which is all we have to work with.

    selected_labelled_idxs = select_new_points_to_label(initial_dataset, v_init)

    save_split_dataset(initial_dataset, selected_labelled_idxs, 0)
    del initial_dataset



#   Now train on the trained dataset for K epochs or until delta train_loss < L
#   Can do this by calling train_masters.py with limited epochs or some special stop condition
#   Or just repeating everything gross


if __name__ == '__main__':
    main()
