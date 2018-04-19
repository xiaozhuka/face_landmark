# -*- coding: utf-8 -*-

"""Some operations.
"""


import numpy as np
import h5py

from tensorpack.dataflow import HDF5Data

from tensorpack.utils import logger

def cf2cl(img):
    """Channel first to channel last.
    """
    i = np.ones([img.shape[1], img.shape[2], img.shape[0]])
    assert img.ndim == 3
    for j in range(img.shape[0]):
        i[:, :, j] = img[j, :, :]
    return


class MyHDF5Dataset(HDF5Data):
    """Because the h5 file was used for caffe, so the image shape is [ns, channels, w, h]
    """
    def __init__(self, filename, data_paths, shuffle=True):
        self.f = h5py.File(filename, 'r')
        logger.info("Loading {} to memory...".format(filename))
        self.dps = []
        for k in data_paths:
            tmp = self.f[k].value
            if tmp.ndim == 4:
                self.dps.append([cf2cl(np.array(tmp[i, :, :, :])) for i in range(tmp.shape[0])])
            else:
                self.dps.append(tmp)
        lens = [len(k) for k in self.dps]
        assert all([k == lens[0] for k in lens])
        self._size = lens[0]
        self.shuffle = shuffle

    def size(self):
        return self._size

    def get_data(self):
        idxs = list(range(self._size))
        if self.shuffle:
            self.rng.shuffle
        for k in idxs:
            yield [dp[k] for dp in self.dps]


def get_dataset():
    """Use tensorpack.dataflow.HDF5Data to get dataset through hdf5 file.

    Warning:
        This will load all data into memory.
    """
    train_path = r'E:\data\ImageList_facepose_25pointsSELECTED_train_ex8_RESIZE50_noFlip_noShuffle.h5'
    test_path = r'E:\data\ImageList_facepose_25pointsSELECTED_test_exMODI8_RESIZE50_noFlip_noShuffle.h5'

    data_paths = ['data', 'landmarks']

    return MyHDF5Dataset(train_path, data_paths), MyHDF5Dataset(test_path, data_paths)