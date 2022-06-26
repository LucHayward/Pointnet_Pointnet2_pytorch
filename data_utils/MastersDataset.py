import os
from line_profiler_pycharm import profile

import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path

import pptk  # For visualisation
from torch import save, load
from time import time

from Visualisation_utils import turbo_colormap_data
rng = np.random.default_rng()


def _adjust_column(column_max, column_min, segment_coord_min, segment_coord_max):
    """
    Adjusts the column boundaries to be within the current segments boundaries
    :param column_max: max (x,y,z,)
    :param column_min: min (x,y,z,)
    :param segment_coord_max: max_boundary_coords (x,y,z,)
    :param segment_coord_min: min_boundary_coords (x,y,z,)
    """
    if column_min[0] < segment_coord_min[0]:
        offset_x = column_min[0] - segment_coord_min[0]
        column_min[0] -= offset_x
        column_max[0] -= offset_x
    if column_min[1] < segment_coord_min[1]:
        offset_y = column_min[1] - segment_coord_min[1]
        column_min[1] -= offset_y
        column_max[1] -= offset_y
    if column_max[0] > segment_coord_max[0]:
        offset_x = column_max[0] - segment_coord_max[0]
        column_min[0] -= offset_x
        column_max[0] -= offset_x
    if column_max[1] > segment_coord_max[1]:
        offset_y = column_max[1] - segment_coord_max[1]
        column_min[1] -= offset_y
        column_max[1] -= offset_y

    assert np.all((column_min >= segment_coord_min[:2],
                   column_max <= segment_coord_max[:2])), "Column bounds outside segment bounds"
    return column_min, column_max


class MastersDataset(Dataset):
    """
    Dataset contains the points for an area (possibly split by train/validation). Access is via a segment_idx
    weighted by the number of points in the segment, i.e. segments with more points will be sampled more times than
    those with fewer points. We do this (rather than simply split the segments up further) to allow for better
    separation during cross validation.
    """

    # @profile
    def __init__(self, split, data_path: Path, num_points_in_block=4096, block_size=1.0, sample_all_points=False,
                 force_even=False):
        """
        Setup the dataset for the heritage data. Expects .npy format XYZIR.
        :param split: {train, validate, test} if you wish to split the data specify the set here and in the pathname of the files
        :param data_path: location of the data files
        :param num_points_in_block: Number of points to be returned when __get_item__() is called
        :param block_size: size of the sampling column
        :param sample_all_points: Whether to sample random columns or the entire segment sequentially.
        """
        assert split in ["train", "validate", "test", None], 'split must be "train", "validate", "test"'
        self.split = split
        self.num_points_in_block = num_points_in_block
        self.block_size = block_size

        self.sample_all_points = sample_all_points
        self.stride = block_size
        self.padding = 0.001

        # if force_even:
        #     self.batch_label_counts = None
        #     self.

        # Given the data_path
        # Load all the segments that are for this split
        segment_paths = sorted(data_path.glob('*.npy'))
        if split is not None:  # if split is None then just load all the .npy files
            segment_paths = [path for path in segment_paths if split in str(path)]
        assert len(segment_paths) > 0, "No segments in path"
        self.segment_points, self.segment_labels = [], []
        self.segment_coord_min, self.segment_coord_max = [], []

        num_points_per_segment = []
        labelweights = np.zeros(2)

        # For each segment load all the points and split the labels out, recording their relative weights
        for path in tqdm(segment_paths):
            xyzir = np.load(path)
            points, labels = xyzir[:, :-1], xyzir[:, -1]
            self.segment_points.append(points)
            self.segment_labels.append(labels)

            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.segment_coord_min.append(coord_min)
            self.segment_coord_max.append(coord_max)
            # assert np.all((np.max(points[:, :3], axis=0) - np.min(points[:, :3], axis=0))[:2] >= block_size), \
            #     "segments smaller than block_size"

            weights, _ = np.histogram(labels, [0, 1, 2])

            labelweights += weights
            num_points_per_segment.append(len(labels))

        # Weights as inverse ratio (ie if labels=[10,20] labelweights = [2,1])
        labelweights = labelweights / np.sum(labelweights)
        labelweights = np.amax(labelweights) / labelweights
        # Cube root of labelweights has log-like effect for when labels are very imbalanced
        self.labelweights = np.power(labelweights, 1 / 3.0)
        self.num_points_per_segment = num_points_per_segment

        if not self.sample_all_points:
            # Sample from each segment based on the relative number of points.
            # Only sets a segment to be sampled if it has at least num_points_per_segment(4096) points int it
            total_points = np.sum(num_points_per_segment)
            sample_probability = num_points_per_segment / total_points  # probability to sample from each segment
            num_iterations = int(
                total_points / num_points_in_block)  # iterations required to sample each point in theory
            segment_idxs = []

            for i in range(len(segment_paths)):
                segment_idxs.extend([i] * int(round(sample_probability[i] * num_iterations)))
            self.segments_idxs = np.array(segment_idxs)
        else:
            # Sample every point in the segment in turn following a grid pattern.
            # Just need to return all the points in one go.

            # self.segments_idxs = np.arange(len(self.segment_points))
            # # First check if a cache exists
            # cache_path_list = list(data_path.glob(f"{split}_all_points.cache"))
            # if len(cache_path_list) > 0:
            #     _num_points_in_block, self.data_segment, self.labels_segment, self.sample_weight_segment, self.point_idxs_segment, \
            #         = load(cache_path_list[0])
            #     if _num_points_in_block == self.num_points_in_block:
            #         returned

            # Concatenate additional segments
            if len(self.segment_points) > 1:
                points = np.vstack(self.segment_points)
                labels = np.hstack(self.segment_labels)
                self.segment_points = [points]
                self.segment_labels = [labels]
            else:
                points = self.segment_points[0]
                labels = self.segment_labels[0]

            self.segments_idxs = np.arange(len(self.segment_points))

            num_points_in_segment = points.shape[0]
            coord_min, coord_max = self.segment_coord_min[0], self.segment_coord_max[0]

            # split the segment into cell grids
            grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
            grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)

            points, grid_mask = self._split_grid_shape(np.hstack((points, labels[:, None])), (grid_x, grid_y))
            points, labels = points[:, :-1], points[:, -1]
            self.segment_points, self.segment_labels = [points], [labels]
            for idx, i in enumerate(np.unique(grid_mask)):
                grid_mask[grid_mask == i] = idx

            data_segment, labels_segment, sample_weight_segment, point_idxs_segment = \
                np.array([]), np.array([]), np.array([]), np.array([])
            return_grid = [[[] for _ in range(grid_y)] for _ in range(grid_x)]
            grid_cell_to_segment = []

            for cell_idx in tqdm(np.unique(grid_mask), desc="Fill batches"):
                point_idxs = np.where(grid_mask == cell_idx)[0]

                # Get batches required
                num_batches = int(np.ceil(point_idxs.size / self.num_points_in_block))

                # Check: May not be necessary to actually pad out the batch like this for inference.
                # If there are not enough points to fill the last batch, set it to replace points.
                point_size = int(num_batches * self.num_points_in_block)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True

                # Duplicate some point_idxs at random from the sample
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]

                # Get Normalized (-1,1) xyz values
                # normlized_xyz = np.zeros((point_size, 3))
                # normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                # normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                # normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]

                #        # Shift XY to start at (0,0)
                #         data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                #         data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                #         data_batch[:, 3:6] /= 255.0

                # data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                # No idea what this is meant to be doing. I think the idea is to get the weighting of the labels in this
                # batch? It's actually getting a weight to assign to each point though.
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                grid_cell_to_segment.append(len(label_batch))
                # Stack all the points/labels from this cell with the previous cells
                data_segment = np.vstack([data_segment, data_batch]) if data_segment.size else data_batch
                labels_segment = np.hstack([labels_segment, label_batch]) if labels_segment.size else label_batch
                sample_weight_segment = np.hstack(
                    [sample_weight_segment, batch_weight]) if labels_segment.size else batch_weight
                point_idxs_segment = np.hstack(
                    [point_idxs_segment, point_idxs]) if point_idxs_segment.size else point_idxs

            # Given all the points/labels reshape them to be returned as self.block_points batches.
            # This DOES mean some of the "blocks" will stretch over the cells.
            self.data_segment = data_segment.reshape((-1, self.num_points_in_block, data_segment.shape[1]))
            self.labels_segment = labels_segment.reshape((-1, self.num_points_in_block))
            self.sample_weight_segment = sample_weight_segment.reshape((-1, self.num_points_in_block))
            self.point_idxs_segment = point_idxs_segment.reshape((-1, self.num_points_in_block))
            self.grid_cell_to_segment = grid_cell_to_segment
            self.grid_mask = grid_mask

            # No need to shuffle, done in the dataloader rather
            # self.random_permutation_idx = np.arange(len(self))
            # np.random.shuffle(self.random_permutation_idx)

            self.grid_mask_segment = np.arange(len(self)).repeat(self.num_points_in_block)

            # save([self.num_points_in_block, self.data_segment, self.labels_segment, self.sample_weight_segment,
            #       self.point_idxs_segment], data_path / f"{split}_all_points.cache")

    def _test_coverage(self, idx: int, iterations):
        """
        Samples the dataset[idx] for n iterations and returns the number of points sampled and returned k times (i.e.
        X number of points sampled K times).
        """
        segment_idx = self.segments_idxs[idx]
        points = self.segment_points[idx]
        labels = self.segment_labels[idx]
        num_points_in_segment = points.shape[0]
        point_sample_cnt = np.zeros(len(points))
        point_returned_cnt = np.zeros(len(points))

        for i in tqdm(range(iterations)):
            # We want at least 1024 points otherwise we should just take everything
            if num_points_in_segment <= 1024:
                print("DEBUG ALERT: num_points_in_segment <= 1024")
                point_idxs = range(num_points_in_segment)
            else:
                # Pick a starting centroid for the column
                center_idx = rng.choice(num_points_in_segment)

                # Adjust the center to be correct
                center = points[center_idx][:2]
                column_min = center - self.block_size / 2.0
                column_max = center + self.block_size / 2.0

                column_min, column_max = _adjust_column(column_max, column_min,
                                                        self.segment_coord_min[idx], self.segment_coord_max[idx])

                assert np.all((column_min >= self.segment_coord_min[idx][:2],
                               column_max <= self.segment_coord_max[idx][:2])), "Column bounds outside segment bounds"

                point_idxs = np.where(  # Get all the points that fall within the column
                    (points[:, 0] >= column_min[0]) & (points[:, 0] <= column_max[0])
                    & (points[:, 1] >= column_min[1]) & (points[:, 1] <= column_max[1])
                )[0]
            point_sample_cnt[point_idxs] += 1

            if len(point_idxs) >= self.num_points_in_block:
                point_idxs = rng.choice(point_idxs, 1024, replace=False)
            else:
                point_idxs = rng.choice(point_idxs, 1024, replace=True)
            point_returned_cnt[point_idxs] += 1

        print("Sampled:\n", np.unique(point_sample_cnt, return_counts=True))
        print("Returned:\n", np.unique(point_returned_cnt, return_counts=True))

        v = pptk.viewer(self.segment_points[0][point_sample_cnt > 0, :3],
                        self.seselfgment_labels[0][point_sample_cnt > 0])
        v_all = pptk.viewer(self.segment_points[0][:, :3], self.segment_labels[0])
        v_returned = pptk.viewer(self.segment_points[0][point_returned_cnt > 0, :3],
                                 self.segment_labels[0][point_returned_cnt > 0])

        return point_sample_cnt, point_returned_cnt

    def _get_item(self, idx: int):
        """
        Return the sampled points (N=self.num_points_in_block) from the segment
        This is a subset of the points in the segment via selecting a random position for a square column
        (sides of length self.block_size) and randomly sampling N points within that column.
        :param idx: index of segment to sample from
        :return: (points, labels,)
        """
        segment_idx = self.segments_idxs[idx]
        points = self.segment_points[segment_idx]
        labels = self.segment_labels[segment_idx]
        num_points_in_segment = points.shape[0]

        # We want at least 1024 points otherwise we should just take everything
        if num_points_in_segment <= 1024:
            print("DEBUG ALERT: num_points_in_segment <= 1024")
            point_idxs = range(num_points_in_segment)
        else:
            if hasattr(self, 'batch_label_counts'):
                num_discard_labels = np.sum(labels)
                discard_label_proportion = num_discard_labels / len(labels)

            # Pick a starting centroid for the column
            center_idx = rng.choice(num_points_in_segment)

            # Adjust the center to be correct
            center = points[center_idx][:2]
            column_min = center - self.block_size / 2.0
            column_max = center + self.block_size / 2.0

            column_min, column_max = _adjust_column(column_max, column_min,
                                                    self.segment_coord_min[segment_idx],
                                                    self.segment_coord_max[segment_idx])

            assert np.all((column_min >= self.segment_coord_min[segment_idx][:2],
                           column_max <= self.segment_coord_max[segment_idx][:2])), \
                f"Column bounds outside segment bounds:\nmin={column_min}\nmax={column_max}"

            point_idxs = np.where(  # Get all the points that fall within the column
                (points[:, 0] >= column_min[0]) & (points[:, 0] <= column_max[0])
                & (points[:, 1] >= column_min[1]) & (points[:, 1] <= column_max[1])
            )[0]

        if len(point_idxs) >= self.num_points_in_block:
            point_idxs = rng.choice(point_idxs, self.num_points_in_block, replace=False)
        else:
            point_idxs = rng.choice(point_idxs, self.num_points_in_block, replace=True)

        # # Get Normalized (-1,1) xyz values
        # normlized_xyz = np.zeros((len(point_idxs), 3))
        # normlized_xyz[:, 0] = points[point_idxs, 0] / self.segment_coord_max[0]
        # normlized_xyz[:, 1] = points[point_idxs, 1] / self.segment_coord_max[1]
        # normlized_xyz[:, 2] = points[point_idxs, 2] / self.segment_coord_max[2]
        # return np.hstack((points[point_idsx],normalized_xyz)), labels[point_idxs]

        return points[point_idxs], labels[point_idxs]

    def _get_item_all(self, idx: int):
        """
        Return the batch sample for a given idx
        """
        return (self.data_segment[idx], self.labels_segment[idx])
                # self.sample_weight_segment[idx], self.point_idxs_segment[idx])
        # return self.data_segment, self.labels_segment, self.sample_weight_segment, self.point_idxs_segment

    def _split_grid_shape(self, points, grid_shape):
        """
        Split the points into a grid pattern
        :param points: (n,7) array of points (XYZIRGB)
        :param grid_shape: (x,y,) tuple of grid grid_shape
        :return: modified points array, grid_mask
        """

        def find_nearest_id(array, value):
            import math
            idx = np.searchsorted(array, value, side="left")
            if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
                return idx - 1
            else:
                return idx

        # Sort and split array along x-axis
        print("Sorting by x axis...", end='')
        stime = time()
        points.view((f'{points.dtype.name},' * points.shape[1])[:-1]).sort(order=['f0'], axis=0)
        print(f"{time()-stime:.2f}s")

        total_distances = points[:, :2].max(axis=0) - points[:, :2].min(axis=0)
        intervals_x = np.asarray(
            [points[:, 0].min(axis=0) + np.ceil(total_distances[0] / grid_shape[0]) * i for i in range(grid_shape[0])][
            1:])
        intervals_y = np.asarray(
            [points[:, 1].min(axis=0) + np.ceil(total_distances[1] / grid_shape[1]) * i for i in range(grid_shape[1])][
            1:])
        interval_idxs_x = [find_nearest_id(points[:, 0], v) for v in intervals_x]

        points = np.array_split(points, interval_idxs_x)

        # Sort and split resulting columns along y-axis
        for i in tqdm(range(len(points)), desc="split y-axis"):
            col = points[i]
            col.view((f'{col.dtype.name},' * col.shape[1])[:-1]).sort(order=['f1'], axis=0)
            interval_idxs_y = [0] + [find_nearest_id(col[:, 1], v) for v in intervals_y] + [col.shape[0]]
            col_grid_mask = np.concatenate(
                [np.repeat(i * grid_shape[1] + j, reps - interval_idxs_y[j - 1]) for j, reps in
                 enumerate(interval_idxs_y[1:], start=1)])
            points[i] = np.hstack((col, col_grid_mask[:, None]))

        points = np.vstack(points)
        return points[:, :-1], points[:, -1].astype(int)

    def get_ouput_format(self):
        """
        Returns the points and labels as a single ndarray of form XYZIR where R is the label
        """
        segment_labels = np.concatenate(self.segment_labels)
        segment_points = np.concatenate(self.segment_points)
        return np.hstack((segment_points, segment_labels[:, None]))

    def __getitem__(self, idx: int):
        """
        Returns a set of sampled points from the segment, or all points if sampling all points.
        """
        if not self.sample_all_points:
            return self._get_item(idx)
        else:
            return self._get_item_all(idx)

    def __len__(self):
        if not self.sample_all_points:
            return len(self.segments_idxs)
        else:
            return self.labels_segment.shape[0]


if __name__ == '__main__':
    from pathlib import Path
    import pptk


    def _test_sample_all_points():
        """
        Loads in a dummy dataset as a sample all points set and tests that all points are sampled correctly.
        """
        dataset = MastersDataset(None, Path('../data/PatrickData/Church/MastersFormat/'),
                                 sample_all_points=True)

        BATCH_SIZE = 16

        all_points_data = []
        all_points_labels = []
        all_points_predictions = []
        for i, grid_data in tqdm(enumerate(dataset), total=len(dataset)):
            # grid_data = data_segment, labels_segment, sample_weight_segment, point_idxs_segment
            available_batches = grid_data[0].shape[0]
            num_batches = int(np.ceil(available_batches / BATCH_SIZE))
            for batch in range(num_batches):
                points, target_labels = grid_data[0][batch * BATCH_SIZE:batch * BATCH_SIZE + BATCH_SIZE], \
                                        grid_data[1][batch * BATCH_SIZE:batch * BATCH_SIZE + BATCH_SIZE]

                # Run inference on data({len(points)} points)
                predictions = np.round(np.random.random(target_labels.size))
                all_points_data.append(points)
                all_points_labels.append(target_labels)
                all_points_predictions.append(predictions)
                # Compare the prediciton with the ground truth
                # Calculate the loss function

        all_points_data = np.vstack(np.vstack(all_points_data))
        all_points_labels = np.hstack(np.vstack(all_points_labels))
        all_points_predictions = np.concatenate(all_points_predictions)
        if (len(np.unique(all_points_data, axis=0)) == np.unique(dataset.segment_points[0], axis=0).shape[0]):
            print(f"Did not sample all points, number of points in dataset" \
                  f"({np.unique(dataset.segment_points[0], axis=0).shape[0]}) != unique sampled points" \
                  f"({np.unique(all_points_data, axis=0)}).")


    print("This shouldn't be run")

    # dataset._test_coverage(0, 400)
    _test_sample_all_points()
    print("Done🎉")
