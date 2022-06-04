import pickle

import numpy as np
import pptk
from pathlib import Path

import yaml
from argparse import Namespace

import Visualisation_utils
import train_masters
from data_utils.MastersDataset import MastersDataset

AL_ITERATION = 0

LOG_DIR = Path("log/active_learning")
FINISHED=False


def save_split_dataset(dataset, selected_points_idxs, dataset_merge=None, points=None, labels=None):
    """
    Splits and saves the dataset out to the log directory (log_dir/AL_ITERATION/)
    :param dataset: dataset being selected from, if None then must provide points and labels
    :param selected_points_idxs: selected train/labelling point indexes
    :param dataset_merge: If there is a dataset to merge the selected points into, otherwise None
    :param points: If no dataset is provided must provide the numpy arrays for these directly
    :param labels: If no dataset is provided must provide the numpy arrays for these directly
    :return:
    """
    save_dir = LOG_DIR / str(AL_ITERATION)
    save_dir.mkdir(exist_ok=True, parents=True)
    if dataset is not None:
        points, labels = dataset.segment_points[0], dataset.segment_labels[0]
    train_points = points[selected_points_idxs]
    train_labels = labels[selected_points_idxs]

    if dataset_merge is not None:
        train_points = np.vstack((dataset_merge.segment_points[0], train_points))
        train_labels = np.hstack((dataset_merge.segment_labels[0], train_labels))
    np.save(save_dir / f"train.npy", np.column_stack((train_points, train_labels[:, None])))

    val_points = Visualisation_utils.numpy_inverse_index(points, selected_points_idxs)
    val_labels = Visualisation_utils.numpy_inverse_index(labels, selected_points_idxs)
    np.save(save_dir / f"validate.npy", np.column_stack((val_points, val_labels[:, None])))


def select_new_points_to_label(dataset, viewer):
    completed_selection = False
    num_grid_cells = len(dataset.grid_cell_to_segment)
    selected_label_idxs, selected_cells = None, None
    while not completed_selection:
        print(f"Select 5% of the cells ({num_grid_cells * .05:.0f}/{num_grid_cells}) for labelling")
        input("Waiting for selection...(enter)")
        selected = viewer.get('selected')
        selected_cells = np.unique(dataset.grid_mask[selected])  # CHECK don't think we need to preserve order here.
        selected_labelled_idxs = np.where(np.in1d(dataset.grid_mask, selected_cells))[0]  # where(grid_mask == selected)
        viewer.set(selected=selected_labelled_idxs)
        print(f"Selected {len(selected_labelled_idxs)} points from {len(selected_cells)} cells "
              f"({len(selected_labelled_idxs) / len(dataset.grid_mask) * 100:.2f}% of points, "
              f"{len(selected_cells) / num_grid_cells * 100:.2f}% of area)")
        completed_selection = input("Happy with selection? To adjust enter N, otherwise enter Y").upper() == "Y"

    return selected_labelled_idxs, selected_cells


def get_high_variance_cells(cell_variance, point_variances, num_cells, grid_mask):
    """
    Choose K segments for labelling by the user (probably the same number as initial selection*3)
    Uses the variance of the cells
    :return: the cells for the high variance points, and the points
    """
    var_cutoff = np.sort(cell_variance)[::-1][num_cells]
    high_var_points_idxs = np.where(point_variances >= var_cutoff)[0]
    num_grid_cells = len(np.unique(grid_mask))
    high_var_cells = np.unique(grid_mask[high_var_points_idxs])
    num_high_var_cells = len(high_var_cells)
    # num_high_var_cells = len(np.unique(val_old_ds.grid_mask[high_var_points_idxs]))
    print(f"Selected high variance cells (mean variance >= {var_cutoff:.2f}), "
          f"{len(high_var_points_idxs)}/{len(point_variances)} points ({len(high_var_points_idxs) / len(point_variances) * 100:.2f}%),"
          f"{num_high_var_cells}/{num_grid_cells} cells ({num_high_var_cells / num_grid_cells * 100:.2f}%)")

    return high_var_cells, high_var_points_idxs
    # return high_var_cells


def get_diverse_cells_by_distance(cells, point_idxs, points, grid_mask, features):
    """
    NOT_YET_IMPLEMENTED
    Given a set of cells (expected high variance set) calculates a diverse subset
    Diversity is computed as the distance measure of the cell from the average of the cells
    TODO could be based on the distance measure of the cell from the average cell computed in feature space
    :param features:
    :return: The cells with high diversity
    """
    from scipy.spatial import distance

    distances = distance.pdist(features, 'sqeuclidean')
    average_cell_position = np.mean(points, axis=0)

    distances = []
    for cell in cells:
        cell_point_idxs = np.where(grid_mask == cell)
        cell_average = np.mean(points(cell_point_idxs))
        distances.append((cell, distance.sqeuclidean(average_cell_position, cell_average)))

    distances = np.array(distances)
    distances.sort()
    # TODO get the top N cells, calculate the point_idxs for those cells and return the cells and the point_idxs

    return None

def get_diversity_ranking(features, variance, n_clusters=10, penalty_factor=0.9):
    """
    Score each sample with its uncertainty U
    Clusters the samples into K clusters based on their feature embeddings
    Sort the regions based on the uncertainty U
    For each region, penalise the scores of the remaining regions in that cluster by some factor P
    The result is a ranking of regions based on uncertainty and diversity
    (such that the most uncertain regions are ranked first, but repeat regions from the same cluster are unlikely).
    """
    from sklearn import cluster
    variance_ordering_idxs = variance.argsort()[::-1]
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(features)

    print(F"Debug: Initial ordering:")
    for idx in variance_ordering_idxs[:20]:
        print(f"Idx {idx}, Cluster {kmeans.labels_[idx]},  variance {variance[idx]:.4f} ")

    adjusted_variance = np.copy(variance)
    for i, idx in enumerate(variance_ordering_idxs):  # iterate over the clusters in order of variance
        current_cluster = kmeans.labels_[idx]
        for x in range(i + 1, len(variance_ordering_idxs)):  # iterate over the remaining clusters in order of variance
            x = variance_ordering_idxs[x]
            if kmeans.labels_[x] == current_cluster:  # Scale the variances in the same cluster
                adjusted_variance[x] *= penalty_factor

    print(f"Old variance_ordering_idxs:\n{list(zip(variance_ordering_idxs[:10], kmeans.labels_[:10]))}")
    adjusted_variance_ordering_idxs = adjusted_variance.argsort()[::-1]
    print(f"New variance_ordering_idxs:\n{list(zip(adjusted_variance_ordering_idxs[:10], kmeans.labels_[variance_ordering_idxs[:10]]))}")
    for idx in adjusted_variance_ordering_idxs[:20]:
        print(f"Idx {idx}, Cluster {kmeans.labels_[idx]},  variance {adjusted_variance[idx]:.4f}")

    return adjusted_variance_ordering_idxs, kmeans.labels_

def generate_initial_data_split():
    # get full pcd
    cache_initial_dataset = Path("data/PatrickData/Church/MastersFormat/cache_full_dataset.pickle")
    initial_dataset = None
    if cache_initial_dataset.exists():
        with open(cache_initial_dataset, "rb") as cache_file:
            initial_dataset = pickle.load(cache_file)
    else:
        initial_dataset = MastersDataset(None, Path("data/PatrickData/Church/MastersFormat"),
                                         sample_all_points=True)
        with open(cache_initial_dataset, "wb") as cache_file:
            pickle.dump(initial_dataset, cache_file)

    v_init = Visualisation_utils.pptk_full_dataset(initial_dataset, include_grid_mask=True, include_intensity=True)
    v_init.color_map(
        Visualisation_utils.turbo_colormap_data[::16] * 16)  # Repeats colours distinguishing adjacent cells.
    v_init.color_map("summer")  # Best for intensity which is all we have to work with.

    selected_labelled_idxs, selected_cells = select_new_points_to_label(initial_dataset, v_init)
    save_split_dataset(initial_dataset, selected_labelled_idxs)
    del initial_dataset

def main():
    global AL_ITERATION
    # generate_initial_data_split()
    for i in range(5):
        #   Now train on the trained dataset for K epochs ((or until delta train_loss < L))
        #   Can do this by calling train_masters.py with limited epochs or some special stop condition
        #   Or just repeating everything gross
        with open(Path(f"configs/train0.yaml"), 'r') as yaml_args:
            train_args = yaml.safe_load(yaml_args)
            train_args = Namespace(**train_args)
        train_args.log_dir = LOG_DIR / 'train'
        train_args.data_path = LOG_DIR / str(AL_ITERATION)
        # train_args.epoch = 1  # We just want to train one epoch for testing
        # train_args.npoint *= 4
        # train_args.batch_size = 8
        # train_args.validate_only = True

        train_masters.main(train_args)

        #   Now we need the predictions from the last good trained model (which we saved in the training)
        with np.load(LOG_DIR / 'train' / 'val_predictions.npz') as npz_file:
            predict_points = npz_file['points']
            predict_preds = npz_file['preds']
            predict_target = npz_file['target']
            predict_variance = npz_file['variance']  # Variances are normalised to [-1,1]
            predict_point_variance = npz_file['point_variance']  # is used
            predict_grid_mask = npz_file['grid_mask'].astype('int16')
            predict_features = npz_file['features']
            predict_samples_per_cell = npz_file['samples_per_cell']

        # Show merged pointcloud in pptk
        # val_old_ds = MastersDataset('validate', LOG_DIR / str(AL_ITERATION), sample_all_points=True)
        # v = pptk.viewer(predict_points[:, :3], predict_points[:, -1], predict_preds, predict_target, predict_point_variance, predict_grid_mask)
        # v.color_map('summer')
        num_cells = 10  # Not sure how many we need, maybe more?
        # high_var_cells, high_var_point_idxs = get_high_variance_cells(predict_variance, predict_point_variance,
        #                                                               len(selected_cells) * 3, predict_grid_mask)
        # diverse_cells = get_diverse_cells_by_distance(high_var_cells, high_var_point_idxs, predict_points, predict_grid_mask, predict_features[high_var_cells])

        adjusted_variance_ordering_idxs, cluster_labels = get_diversity_ranking(predict_features, predict_variance, num_cells)
        new_point_idxs = np.where(np.in1d(predict_grid_mask, adjusted_variance_ordering_idxs[:num_cells]))[0]
        # v_new = pptk.viewer(predict_points[new_point_idxs, :3], predict_points[new_point_idxs, -1], predict_preds[new_point_idxs], predict_target[new_point_idxs], predict_point_variance[new_point_idxs], predict_grid_mask[new_point_idxs])
        # v.set(selected=new_point_idxs)

        train_dataset = MastersDataset("train", LOG_DIR / str(AL_ITERATION))

        AL_ITERATION += 1
        save_split_dataset(None, new_point_idxs, train_dataset, predict_points, predict_target)


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
