import os

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

# import pptk

rng = np.random.default_rng()


class S3DISDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0,
                 sample_rate=1.0, transform=None, num_classes=13):
        """

        Args:
            split:
            data_root:
            num_point: Number of points to return
            test_area: Which AREA in the dataset to reserve as the test area
            block_size: Size of a square column (z=0) as side length
            sample_rate:
            transform:
        """
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.num_classes = num_classes
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []  # number of points per room
        labelweights = np.zeros(num_classes)  # Count of labels across all rooms

        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:3], room_data[:, -1]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(num_classes + 1))  # count of class labels in room
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)  # Labelweights = proportion of each label
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(f'Labelweights={self.labelweights}')
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    # @profile
    def __getitem__(self, idx):
        """
        Given a room ID returns self.num_point points, labels
        Args:
            idx ():

        Returns: points (Global XYZ, IGB/255, XYZ/max(room_XYZ)), labels

        """
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # N * 6
        labels = self.room_labels[room_idx]  # N
        N_points = points.shape[0]
        # print(f"DEBUG: labelHist = {np.histogram(labels, [0, 1, 2])}")
        # print(f"DEBUG: AvailablePoints/numPoints = {labels.size}/{self.num_point}={labels.size / self.num_point}")
        # TODO more even class sampling, maybe
        # DEBUG: v = pptk.viewer(points[:,:3],labels)
        # TODO: try this https://ethankoch.medium.com/incredibly-fast-random-sampling-in-python-baf154bd836a
        while (True):  # Repeat until there are at least 1024 point_idxs selected
            if self.num_classes == 2 and 1 in labels:
                tmp = np.arange(len(labels))[labels == 1]
                center_idx = tmp[np.random.randint(0, len(tmp))]
                if labels[center_idx] != 1:
                    print("PROBLEM<>PROBLEM<>PROBLEM<>PROBLEM<>PROBLEM<>PROBLEM<>PROBLEM<>PROBLEM<>PROBLEM")
            else:
                center_idx = np.random.choice(N_points)
            center = points[center_idx][:3]  # Pick random point as center
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0,
                                  0]  # Get a square column (z=0) of size block_size
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (
                        points[:, 1] <= block_max[1]))[0]  # Get all points that fall within the square column
            # print(f'DEBUG: Center Column Hist = {np.histogram(labels[point_idxs], [0, 1, 2])}')
            # DEBUG: v = pptk.viewer(points[point_idxs,:3],labels[point_idxs])
            if point_idxs.size > 1024:
                break
            else:
                self.block_size *= 2
                print(f'DEBUG: increasing block size to {self.block_size}')

        if point_idxs.size >= self.num_point:  # Select points from the point_idxs up until self.num_point, with replacement if necessary\
            # TODO Fix this shit
            if self.num_classes == 2:
                discard_point_mask = np.bool_(labels[point_idxs])
                discard_point_idxs = point_idxs[discard_point_mask]
                keep_point_idxs = point_idxs[~discard_point_mask]

                if len(discard_point_idxs) >= self.num_point // 2 and len(keep_point_idxs) >= self.num_point // 2:
                    selected_point_idxs = np.concatenate((np.random.choice(discard_point_idxs, self.num_point // 2,
                                                                           replace=False),
                                                          np.random.choice(keep_point_idxs, self.num_point // 2,
                                                                           replace=False)))  # TODO Look into changing the sampling here
                elif len(keep_point_idxs) <= self.num_point // 2:
                    selected_point_idxs = np.concatenate((np.array(keep_point_idxs),
                                                          np.random.choice(discard_point_idxs,
                                                                           self.num_point - len(keep_point_idxs),
                                                                           replace=False)))
                else:
                    selected_point_idxs = np.concatenate((np.array(discard_point_idxs),
                                                          np.random.choice(keep_point_idxs,
                                                                           self.num_point - len(discard_point_idxs),
                                                                           replace=False)))
            else:
                selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # print(f"DEBUG: Selected_labelHist = {np.histogram(labels[selected_point_idxs], [0, 1, 2])}")

        # 05/06/2022
        # selected_points => XYZRGB
        # current_points[:,6:9] => xyz/max(room) approximately normalized [-1,1]
        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros(
            (self.num_point, 9))  # num_point * 9 (last three store XYZ/max(room_XYZ) aka Normalised) CHECK not true now
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        # selected_points_idxs[:, 0] = selected_points_idxs[:, 0] - center[0]  # Translate XY so last center is at origin
        # selected_points_idxs[:, 1] = selected_points_idxs[:, 1] - center[1]
        # selected_points_idxs[:, 3:6] /= 255.0 #TODO Fix this
        current_points[:, 0:3] = selected_points  # Global XYZ, IGB/255, XYZ/max(room_XYZ)
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        # pptk.viewer(np.concatenate((current_points[:,:3], current_points[:,6:],generate_bounding_wireframe_points(current_points[:,:3].min(axis=0), current_points[:,:3].max(axis=0),50)[0],generate_bounding_wireframe_points(current_points[:,6:].min(axis=0), current_points[:,6:].max(axis=0),50)[0], generate_bounding_cube([0,0,0],1)[0])), np.concatenate((current_labels+2,current_labels,generate_bounding_wireframe_points(current_points[:,:3].min(axis=0), current_points[:,:3].max(axis=0),50)[1][:,0], generate_bounding_wireframe_points(current_points[:,6:].min(axis=0), current_points[:,6:].max(axis=0),50)[1][:,0], generate_bounding_cube([0,0,0],1)[1][:,0])))
        return current_points, current_labels, room_idx

    def __len__(self):
        return len(self.room_idxs)


class ScannetDatasetWholeScene():
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', test_area=5, stride=0.5, block_size=1.0, padding=0.001,
                 num_classes=13):
        self.block_points = block_points #num_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('Area_%d' % test_area) is not -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(num_classes)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(num_classes + 1))
            # print(f"DEBUG: Loading data seg with hist = {tmp}")
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        # print(f"DEBUG: labelweights = {labelweights}")
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:, :6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]), np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                # For each cell in the grid
                # Get the start/end coords of the cell
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                # Get all the points within the cell (or continue if empty)
                point_idxs = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (
                            points[:, 1] >= s_y - self.padding) & (
                            points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue
                # Get batches required
                num_batches = int(np.ceil(point_idxs.size / self.block_points))
                # If there are not enough points to fill the last batch, set it to replace points.
                point_size = int(num_batches * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True

                # add on some extra point_idxs and shuffle them.
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]

                # Get Normalized (-1,1) xyz values
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]

                # Shift XY to start at (0,0)
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

                # Stack all the points/labels from this cell with the previous cells
                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs

        # Given all the points/labels reshape them to be returned as self.block_points batches.
        # This DOES mean some of the "blocks" will stretch over the cells.
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)


if __name__ == '__main__':
    data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area,
                              block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random

    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)


    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)


    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()
