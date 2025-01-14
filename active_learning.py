import argparse
import importlib
import pickle

import numpy as np
import pptk
from pathlib import Path
from sklearn.metrics import accuracy_score, jaccard_score

import yaml
from argparse import Namespace

from tqdm import tqdm

import Visualisation_utils
import train_masters
from data_utils.MastersDataset import MastersDataset

AL_ITERATION = 0
GROUP_NAME = None
NOTES = None

LOG_DIR = Path("log/active_learning")
FINISHED = False

MERGED_ACCURACY = []
MERGED_IOU = []
MERGED_TRAIN_PERCENTAGE = []
MERGED_VAL_PERCENTAGE = []


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


def select_new_points_to_label(dataset, viewer, proportion_cells=0.05):
    completed_selection = False
    num_grid_cells = len(dataset.grid_cell_to_segment)
    args.label_budget = int(num_grid_cells * args.label_budget)
    selected_label_idxs, selected_cells = None, None
    while not completed_selection:
        print(
            f"Select {proportion_cells * 100}% of the cells ({num_grid_cells * proportion_cells:.0f}/{num_grid_cells}) for labelling")
        input("Waiting for selection...(enter)")
        selected = viewer.get('selected')
        selected_cells = np.unique(dataset.grid_mask[selected])  # CHECK don't think we need to preserve order here.
        selected_labelled_idxs = np.where(np.in1d(dataset.grid_mask, selected_cells))[0]  # where(grid_mask == selected)
        viewer.set(selected=selected_labelled_idxs)
        print(f"Selected {len(selected_labelled_idxs)} points from {len(selected_cells)} cells "
              f"({len(selected_labelled_idxs) / len(dataset.grid_mask) * 100:.2f}% of points, "
              f"{len(selected_cells) / num_grid_cells * 100:.2f}% of area)\n" #TODO change area to volume, 
              f"DEBUG: keep:discard = {np.unique(dataset.segment_labels[0][selected_labelled_idxs], return_counts=True)[1] / len(selected_labelled_idxs) * 100}")
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


def round_to_N_ignoring_leading_zeros(number, ending_values_to_keep=4):
    """
    rounds the number off to N trailing decimals, ignoring leading values (including 0)
    1.000016 -> 1.00001600
    0.000016 -> 0.00001600
    11.0000123456789 -> 11.00001235

    :param ending_values_to_keep:
    :param number:
    :return:
    """
    # TODO this doesn't handle when tere are only zeros
    import re
    altered = False
    if number < 1:
        altered = True
        number += 1
    regex = r"0[1-9]"
    float_part = str(number).split(".")[1]
    float_limiter = re.search(regex, float_part).start() if float_part.startswith("0") else -1
    if altered:
        number -= 1
    return eval(f'{number:2.{float_limiter + 1 + ending_values_to_keep}f}')


def get_diversity_ranking(features, uncertainty, n_clusters=10, penalty_factor=0.9):
    """
    Score each sample with its uncertainty U
    Clusters the samples into K clusters based on their feature embeddings
    Sort the regions based on the uncertainty U
    For each region, penalise the scores of the remaining regions in that cluster by some factor P
    The result is a ranking of regions based on uncertainty and diversity
    (such that the most uncertain regions are ranked first, but repeat regions from the same cluster are unlikely).
    :param penalty_factor: How much to reduce the weighting of subsequent cells in a cluster (1 = no change, 0 = remove)
    :return: adjusted uncertainty ordering idxs,
    """
    from sklearn import cluster
    uncertainty_ordering_idxs = uncertainty.argsort()[::-1]

    # Find idxs for zero uncertainty cells and shuffle those
    # TODO shuffle all within unique buckets
    b = None
    for i, x in enumerate(uncertainty_ordering_idxs):
        if uncertainty[x] == 0:
            b = i
            break
    if b is not None:
        np.random.shuffle(uncertainty_ordering_idxs[b:])

    # CHECK is normalizing this the same as cosine?
    # if args.normalize_feats:
    #     from sklearn.preprocessing import normalize
    #     features = normalize(features)
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(features)

    cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
    for c, n in zip(cluster_ids, cluster_sizes):
        print(f"Clusters {c}: {n} cells")

    print(F"Debug: Initial ordering ({len(uncertainty_ordering_idxs)} cells):")
    for idx in uncertainty_ordering_idxs[:20]:
        print(
            f"Idx {idx}, Cluster {kmeans.labels_[idx]},  uncertainty {uncertainty[idx]:.5g} ")

    adjusted_uncertainty = np.copy(uncertainty)
    for i, idx in enumerate(uncertainty_ordering_idxs):  # iterate over the clusters in order of uncertainty
        current_cluster = kmeans.labels_[idx]
        for x in range(i + 1,
                       len(uncertainty_ordering_idxs)):  # iterate over the remaining clusters in order of uncertainty
            x = uncertainty_ordering_idxs[x]
            if kmeans.labels_[x] == current_cluster:  # Scale the uncertaintys in the same cluster
                adjusted_uncertainty[x] *= penalty_factor

    print(
        f"Old uncertainty_ordering_idxs:\n"
        f"{list(zip(uncertainty_ordering_idxs[:10], kmeans.labels_[uncertainty_ordering_idxs[:10]]))}")
    adjusted_uncertainty_ordering_idxs = adjusted_uncertainty.argsort()[::-1]
    print(
        f"New uncertainty_ordering_idxs:\n"
        f"{list(zip(adjusted_uncertainty_ordering_idxs[:10], kmeans.labels_[adjusted_uncertainty_ordering_idxs[:10]]))}")
    for idx in adjusted_uncertainty_ordering_idxs[:20]:
        print(
            f"Idx {idx}, Cluster {kmeans.labels_[idx]},  uncertainty {adjusted_uncertainty[idx]:.5g}")

    return adjusted_uncertainty_ordering_idxs, kmeans.labels_


def generate_initial_data_split(initial_labelling_budget,
                                init_dataset_path="data/PatrickData/Church/MastersFormat"):
    init_dataset_path = Path(init_dataset_path)
    # get full pcd
    cache_initial_dataset = init_dataset_path / 'cache_full_dataset.pickle'
    initial_dataset = None
    if cache_initial_dataset.exists():
        with open(cache_initial_dataset, "rb") as cache_file:
            initial_dataset = pickle.load(cache_file)
    else:
        initial_dataset = MastersDataset(None, init_dataset_path,
                                         sample_all_points=True)
        with open(cache_initial_dataset, "wb") as cache_file:
            pickle.dump(initial_dataset, cache_file)

    v_init = Visualisation_utils.pptk_full_dataset(initial_dataset, include_grid_mask=True, include_intensity=True)
    v_init.color_map(
        Visualisation_utils.turbo_colormap_data[::16] * 16)  # Repeats colours distinguishing adjacent cells.
    v_init.color_map("summer")  # Best for intensity which is all we have to work with.

    selected_labelled_idxs, selected_cells = select_new_points_to_label(initial_dataset, v_init,
                                                                        initial_labelling_budget)
    save_split_dataset(initial_dataset, selected_labelled_idxs)
    del initial_dataset


def log_merged_metrics(train_labels, predict_preds, predict_target):
    """
    Given a training dataset and the prediction/target labels for the remaining data calculate the accuracy and mIoU
    Append this to the global lists
    :param train_labels:
    :param predict_preds:
    :param predict_target:
    :return:
    """
    merged_preds = np.hstack((train_labels, predict_preds))
    merged_target = np.hstack((train_labels, predict_target))

    accuracy = accuracy_score(y_true=merged_target, y_pred=merged_preds)
    IoU = jaccard_score(merged_target, merged_preds)
    mIoU = jaccard_score(merged_target, merged_preds, average='macro')

    # accuracy = sum(merged_preds == merged_target) / len(merged_target)
    #
    # IoU, mIoU = calculate_iou(merged_preds, merged_target)

    MERGED_ACCURACY.append(accuracy)
    MERGED_IOU.append(mIoU)


def calculate_iou(preds, target):
    """
    Deprecated, user sklearn.metrics.jaccard_score(target,preds)
    Calculates the iou and mIoU for a binary classifier give the predictions and the target labels
    :param preds: the predicted class labels
    :param target:  the target class labels
    :return: the IoU of each class and the meanIoU
    """
    IoU = jaccard_score(target, preds)
    mIoU = jaccard_score(target, preds, average='true')

    # total_seen_class, total_correct_class, total_iou_denominator_class = [0, 0], [0, 0], [0, 0]
    # for l in range(2):
    #     target_l = (target == l)
    #     pred_l = (preds == l)
    #
    #     total_seen_class[l] += np.sum(target_l)  # How many times the label was available
    #     # How often the predicted label was correct in the batch
    #     total_correct_class[l] += np.sum(pred_l & target_l)
    #     # Total predictions + Class occurrences (Union prediction of class (right or wrong) and actual class occurrences.)
    #     total_iou_denominator_class[l] += np.sum((pred_l | target_l))
    #
    # IoU = np.array(total_correct_class) / (np.array(total_iou_denominator_class,
    #                                                 dtype=np.float64) + 1e-6)  # correct prediction/class occurrences + false prediction
    # mIoU = np.mean(IoU)
    return IoU, mIoU


def main(args):
    global AL_ITERATION
    if args.model == 'pointnet++':
        MODEL = importlib.import_module("train_masters")
    elif args.model == "RF":
        MODEL = importlib.import_module("train_rf")
    elif args.model == "KPConv":
        raise NotImplementedError

    num_AL_loop = 6
    MERGED_TRAIN_PERCENTAGE=[args.init_label_budget*100]
    for i in range(num_AL_loop-1):
        MERGED_TRAIN_PERCENTAGE.append(MERGED_TRAIN_PERCENTAGE[-1]+args.label_budget*100)
    generate_initial_data_split(initial_labelling_budget=args.init_label_budget, init_dataset_path=args.data_path)
    for i in tqdm(range(num_AL_loop), desc="AL Loop"):
        AL_ITERATION = i

        # Setup the wandb logging using group names inside the loop so that you can track the runs
        # as several lines on the same plot
        wandb.init(project="Masters", name=f'{GROUP_NAME}_{i}', group=GROUP_NAME,
                   notes=NOTES)
        if i == 0: wandb.run.log_code(".")

        if args.model == 'pointnet++':
            with open(Path(f"configs/pointnet++/lowerWDepochs12.yaml"), 'r') as yaml_args:
                train_args = yaml.safe_load(yaml_args)
                train_args = Namespace(**train_args)
            train_args.log_dir = LOG_DIR / str(AL_ITERATION) / 'train'
            train_args.data_path = LOG_DIR / str(AL_ITERATION)
            train_args.data_path.mkdir(exist_ok=True, parents=True)
            train_args.log_dir.mkdir(exist_ok=True, parents=True)

            # Move the best_train_model from the previous iteration to this iterations log_dir
            if AL_ITERATION > 0:
                import shutil
                checkpoint_dir = (train_args.log_dir / 'checkpoints')
                checkpoint_dir.mkdir(exist_ok=True, parents=True)
                old_best_model = LOG_DIR / str(AL_ITERATION - 1) / 'train/checkpoints/best_train_model.pth'
                shutil.copy(old_best_model, checkpoint_dir / 'best_model.pth')
            else:
                train_args.epoch = 6

            # train_args.epoch = 20  # set in config yaml
            # train_args.npoint *= 4
            # train_args.batch_size = 8
            # train_args.validate_only = True


        elif args.model == 'RF':
            with open(Path(f"configs/RandomForests/default.yaml")) as yaml_args:
                train_args = yaml.safe_load(yaml_args)
                train_args = Namespace(**train_args)
            train_args.log_dir = LOG_DIR / str(AL_ITERATION) / 'train'
            train_args.data_path = LOG_DIR / str(AL_ITERATION)
            train_args.data_path.mkdir(exist_ok=True, parents=True)
            train_args.log_dir.mkdir(exist_ok=True, parents=True)

        print(f"--- running training loop {i} ---")
        wandb.config.update(train_args)
        MODEL.main(wandb.config)
        print(f"--- finished training loop {i} ---")

        #   Now we need the predictions from the last good trained model (which we saved in the training)
        with np.load(LOG_DIR / str(AL_ITERATION) / 'train' / 'val_predictions.npz') as npz_file:
            predict_points = npz_file['points']
            predict_preds = npz_file['preds']
            predict_target = npz_file['target']
            predict_variance = npz_file['variance']  # Variances are normalised to [-1,1]
            predict_point_variance = npz_file['point_variance']  # was used
            predict_grid_mask = npz_file['grid_mask'].astype('int16')
            predict_features = npz_file['features']
            # predict_samples_per_cell = npz_file['samples_per_cell']

        # Show merged pointcloud in pptk
        # val_old_ds = MastersDataset('validate', LOG_DIR / str(AL_ITERATION), sample_all_points=True)
        # v = pptk.viewer(predict_points[:, :3], predict_points[:, -1], predict_preds, predict_target, predict_point_variance, predict_grid_mask)
        # v.color_map('summer')
        # high_var_cells, high_var_point_idxs = get_high_variance_cells(predict_variance, predict_point_variance,
        #                                                               len(selected_cells) * 3, predict_grid_mask)
        # diverse_cells = get_diverse_cells_by_distance(high_var_cells, high_var_point_idxs, predict_points, predict_grid_mask, predict_features[high_var_cells])

        adjusted_uncertainty_ordering_idxs, cluster_labels = get_diversity_ranking(predict_features, predict_variance,
                                                                                   args.num_clusters)
        new_point_idxs = \
            np.where(np.in1d(predict_grid_mask, adjusted_uncertainty_ordering_idxs[:args.label_budget]))[0]
        # v_new = pptk.viewer(predict_points[new_point_idxs, :3], predict_points[new_point_idxs, -1], predict_preds[new_point_idxs], predict_target[new_point_idxs], predict_point_variance[new_point_idxs], predict_grid_mask[new_point_idxs])
        # v.set(selected=new_point_idxs)

        train_dataset = MastersDataset("train", LOG_DIR / str(AL_ITERATION))
        log_merged_metrics(train_dataset.segment_labels[0], predict_preds, predict_target)

        AL_ITERATION += 1
        save_split_dataset(None, new_point_idxs, train_dataset, predict_points, predict_target)
        wandb.finish()

    wandb.init(project="Masters", name=f'{GROUP_NAME}_summary', group=GROUP_NAME,
               notes=NOTES)
    for acc, iou, train_p, val_p in zip(MERGED_ACCURACY, MERGED_IOU, MERGED_TRAIN_PERCENTAGE, MERGED_VAL_PERCENTAGE):
        wandb.log({'merged_accuracy': acc, 'merged_mIoU': iou, 'train_percentage': train_p, 'val_percentage': val_p})


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='RF', help="Which model (pointnet++, RF, KPConv)",
                        choices=["pointnet++", "RF", "KPConv"])
    parser.add_argument('--init_label_budget', default=0.05, help="Initial labelling budget as a fraction of cells")
    parser.add_argument('--label_budget', default=0.05, help="Labelling budget after each AL iteration")
    parser.add_argument('--num_clusters', default=75, help='Number of clusters for KMeans')
    parser.add_argument('--distance_metric', default='cosine', help='Distance metric for clustering in feature space')
    parser.add_argument('--data_path', default='data/PatrickData/Church/MastersFormat', help='Data path from root')

    return parser.parse_args()



if __name__ == '__main__':
    import os
    import wandb

    args = parse_args()

    os.environ["WANDB_MODE"] = "dryrun"
    args.label_budget = 19

    # GROUP_NAME = f'AL-{args.model}: WD1e-2_5%_repeat10epochs12'
    # NOTES = "WD 1e-2 Pointnet++ 5% area, 10 repeats, 12epoch"
    # LOG_DIR = LOG_DIR / GROUP_NAME
    # LOG_DIR.mkdir(parents=True, exist_ok=True)

    main(args)
