import numpy as np
from pathlib import Path


def is_array_sorted(arr):
    """Is a numpy array in ascending sorted order"""
    return np.all(arr[:-1] <= arr[1:])


def order(pcd):
    ind = np.lexsort((t[:, 2], t[:, 1], t[:, 0]))
    if not is_array_sorted(ind):
        pcd = pcd[ind]
    return pcd


if __name__ == '__main__':
    data_path = Path(
        '/home/luc/PycharmProjects/Pointnet_Pointnet2_pytorch/data/PatrickData/Church/MastersFormat/hand_selected_reversed')
    t, v = np.load(data_path / 'train.npy'), np.load(data_path / 'validate.npy')
    t, v = order(t), order(v)
    np.save(data_path / 'train.npy', t), np.save(data_path / 'validate.npy', v)

