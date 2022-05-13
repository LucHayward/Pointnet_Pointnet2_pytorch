import pickle

import numpy as np
import torch
import pptk
import open3d as o3d
from pathlib import Path

import yaml
from argparse import Namespace

import Visualisation_utils
import train_masters
from data_utils.MastersDataset import MastersDataset

TEMP_DIR = Path("temp/")
LOG_DIR = Path("log/active_learning")


def save_split_dataset(dataset, selected_points, iteration: int, dataset_merge=None):
    """
    Splits and saves the dataset out to the log directory (log_dir/iteration/)
    TODO change to the log dir
    :param dataset: dataset being selected from
    :param selected_points: selected train/labelling points
    :param iteration: which iteration of the Active Learning process
    :param dataset_merge: If there is a dataset to merge the selected points into, otherwise None
    :return:
    """
    train_points = dataset.segment_points[0][selected_points]
    train_labels = dataset.segment_labels[0][selected_points]
    if dataset_merge is not None:
        train_points = np.vstack((dataset_merge.segment_points[0][selected_points], train_points))
        train_labels = np.hstack((dataset_merge.segment_labels[0][selected_points], train_labels))
    np.save(LOG_DIR / str(iteration) / f"train.npy", np.column_stack((train_points, train_labels[:, None])))

    val_points = Visualisation_utils.numpy_inverse_index(dataset.segment_points[0], selected_points)
    val_labels = Visualisation_utils.numpy_inverse_index(dataset.segment_labels[0], selected_points)
    np.save(LOG_DIR / str(iteration) / f"validate.npy", np.column_stack((val_points, val_labels[:, None])))


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
    # TEMP_DIR.mkdir(exist_ok=True)
    # get full pcd
    cache_initial_dataset = Path("data/PatrickData/Church/MastersFormat/cache_full_dataset.pickle")
    initial_dataset = None
    if cache_initial_dataset.exists():
        with open(cache_initial_dataset, "rb") as cache_file:
            initial_dataset = pickle.load(cache_file)
    else:
        initial_dataset = MastersDataset(None, Path("data/PatrickData/Church/MastersFormat"), sample_all_points=True)
        with open(cache_initial_dataset, "wb") as cache_file:
            pickle.dump(initial_dataset, cache_file)

    v_init = Visualisation_utils.pptk_full_dataset(initial_dataset, include_grid_mask=True, include_intensity=True)
    v_init.color_map(
        Visualisation_utils.turbo_colormap_data[::16] * 16)  # Repeats colours distinguishing adjacent cells.
    v_init.color_map("summer")  # Best for intensity which is all we have to work with.

    selected_labelled_idxs = select_new_points_to_label(initial_dataset, v_init)
    AL_iteration = 0
    save_split_dataset(initial_dataset, selected_labelled_idxs, AL_iteration)
    del initial_dataset

    #   Now train on the trained dataset for K epochs ((or until delta train_loss < L))
    #   Can do this by calling train_masters.py with limited epochs or some special stop condition
    #   Or just repeating everything gross
    train_args = None
    with open(Path("configs/train0.yaml"), 'r') as yaml_args:
        train_args = yaml.safe_load(yaml_args)
        train_args = Namespace(**train_args)
    train_args.log_dir = LOG_DIR / 'train'
    train_args.data_path = LOG_DIR / str(AL_iteration)
    train_args.epoch = 1  # We just want to train one epoch for testing
    train_masters.main(train_args)


if __name__ == '__main__':
    import os
    import wandb

    os.environ["WANDB_MODE"] = "dryrun"
    run_name = 'AL: testing'
    LOG_DIR = LOG_DIR / run_name
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    wandb.init(project="Masters", resume=False, name=run_name,
               notes="Testing the active learning wrappers by running train_masters.main() directly.")
    wandb.run.log_code(".")

    main()
