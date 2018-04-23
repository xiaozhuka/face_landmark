# -*- coding: utf-8 -*-

"""Some operations.
"""


import numpy as np
import h5py
import tensorflow as tf
from random import random
from skimage.transform import resize, rotate

import os
from math import floor, ceil
from cv2 import imread, imwrite
from dlib_face_detection import dlib_file_list
from config import ModelConfig

import math

import cv2

path=[r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\25points_selected',
            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\ce',
            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\hu',
            r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\low']

def cf2cl(img):
    """Channel first to channel last.
    """
    i = np.ones([img.shape[1], img.shape[2], img.shape[0]])
    assert img.ndim == 3
    for j in range(img.shape[0]):
        i[:, :, j] = img[j, :, :]
    return i



class MyHDF5Dataset():
    """Because the h5 file was used for caffe, so the image shape is [ns, channels, h, w]
    """
    def __init__(self, filename, data_paths):
        self.f = h5py.File(filename, 'r')
        self.data_paths = data_paths
        self.dps = [self.f[k].value for k in data_paths]
        for tmp in range(len(self.dps)):
            if self.dps[tmp].ndim == 4:
                tmp_list = [cf2cl(self.dps[tmp][img, ...]) for img in range(self.dps[tmp].shape[0])]
                self.dps[tmp] = np.array(tmp_list)
            else:
                continue
        lens = [len(k) for k in self.dps]
        assert all([k == lens[0] for k in lens])
        self._size = lens[0]

    def __len__(self):
        return self._size

    @property
    def data(self):
        return [self.dps[i] for i in range(len(self.data_paths))]

def read_fn(imgs, labels, config):
    i = tf.constant(imgs)
    l = tf.constant(labels)
    dataset = tf.contrib.data.Dataset.from_tensor_slices((i, l))

    # dataset = dataset.map(
    #     lambda f,l: tf.py_func(read_py_func, [f, l], [tf.float32, tf.float32]))
    # dataset = dataset.map(lambda f, l: _preprocess_tf_func(f, l, config), num_thread=32)
    return dataset


def test_dataset(ds):
    iterator = ds.make_one_shot_iterator()
    next_element = iterator.get_next()
    sess = tf.Session()
    return sess.run(next_element)


def random_augmentation(scale=None,
                        dirty_circle=0.5,
                        random_flip_horizontal=0.5,
                        random_filp_vertical=0.5,
                        random_noise=None,
                        random_landmark_mask=0.5,
                        random_rotate=0.5,
                        random_squeeze=[0.5, 6],
                        test=False):
    """Random augmentate dataset wrapper for parameters.
    Params:
        scale: [probability of scale, ratio of scale]
    """
    assert random_noise is None, "Not supported random_noise"
    # assert scale is None, "Not supported scale"
    def random_augmentation_func(ds):
        """
        Params:
            ds: [imgs, labels]
        """
        assert ds[0].ndim == 3, "Dataset first element must be images, 3 dim ndarray"

        img = ds[0]
        tmp_img = img.copy()
        tmp_lab = ds[1]
        shape_ori = tmp_img.shape

        # random scale
        if scale is not None:
            if random() < scale[0]:
                tmp_zeros = np.zeros(shape_ori)
                shape_aft = [ floor(tmp * (1.0 - 1.0/scale[1])) for tmp in shape_ori[:2] ]
                tmp_img = resize(tmp_img, shape_aft)
                tmp_lab[::2] = tmp_lab[::2] * (1.0 - 1.0 / scale[1])
                tmp_lab[1::2] = tmp_lab[1::2] * (1.0 - 1.0 / scale[1])

                base_x = floor(random() * (shape_ori[0] - shape_aft[0]))
                base_y = floor(random() * (shape_ori[1] - shape_aft[1]))

                tmp_lab[::2] = tmp_lab[::2] + base_y
                tmp_lab[1::2] = tmp_lab[1::2] + base_x

                tmp_zeros[base_x:(shape_aft[0]+base_x), base_y:(shape_aft[1]+base_y)] = tmp_img
                tmp_img = tmp_zeros

        if random_squeeze is not None:
            if random() < random_squeeze[0]:
                tmp_zeros = np.zeros(shape_ori)
                shape_aft = [floor(tmp * (1.0 - 1.0 / random_squeeze[1])) for tmp in shape_ori[:2]]
                if random() > 0.5:
                    dir = 'x'
                    shape_aft[1] = shape_ori[1]
                    tmp_lab[1::2] = tmp_lab[1::2] * (1.0 - 1.0 / random_squeeze[1])
                else:
                    dir = 'y'
                    shape_aft[0] = shape_ori[0]
                    tmp_lab[0::2] = tmp_lab[0::2] * (1.0 - 1.0 / random_squeeze[1])

                tmp_img = resize(tmp_img, shape_aft)
                tmp_zeros[0:shape_aft[0], 0:shape_aft[1]] = tmp_img
                tmp_img = tmp_zeros

        if dirty_circle is not None:
            if random() < dirty_circle:
                # add some mask
                tmp_for_shape_choice = random()
                rand_color = [np.random.randint(10, 245) for _ in range(3)]
                if tmp_for_shape_choice > 0.5:
                    rand_r = random() * min(shape_ori[:2]) / 8.0
                    rand_x = random() * (min(shape_ori[:2]) - rand_r - 1)
                    rand_y = random() * (min(shape_ori[:2]) - rand_r - 1)
                    cv2.circle(tmp_img, (int(rand_x), int(rand_y)), int(rand_r), rand_color, -1)
                else:
                    rand_w = int(random() * (min(shape_ori[:2]) / 4.0))
                    rand_x = int(random() * (min(shape_ori[:2]) - rand_w - 1))
                    rand_y = int(random() * (min(shape_ori[:2]) - rand_w - 1))
                    cv2.rectangle(tmp_img, (rand_x, rand_y), (rand_x+rand_w, rand_y+rand_w), rand_color, -1)

        if random_rotate is not None:
            if random() < random_rotate:
                tmp_img[tmp_img > 1] = 1
                tmp_angle = (random() - 0.5) * 20
                tmp_img = rotate(tmp_img, angle=tmp_angle)
                center = np.array(tmp_img.shape[:2]) / 2 - 0.5
                for tmp_idx in range(int(len(tmp_lab) / 2)):
                    x = tmp_lab[tmp_idx*2] - center[0]
                    y = tmp_lab[tmp_idx*2+1] - center[1]
                    tmp_lab[tmp_idx*2] = x*math.cos(tmp_angle*3.1416/180) + y*math.sin(tmp_angle*3.1416/180) + center[0]
                    tmp_lab[tmp_idx*2+1] = -x*math.sin(tmp_angle*3.1416/180) + y*math.cos(tmp_angle*3.1416/180) + center[1]

        if random_filp_vertical is not None:
            if random() < random_filp_vertical:
                tmp_img = np.flipud(tmp_img) # up down
                tmp_lab[1::2] = shape_ori[0] - tmp_lab[1::2]

        if random_flip_horizontal is not None:
            if random() < random_flip_horizontal:
                tmp_img = np.fliplr(tmp_img)  # up down
                tmp_lab[::2] = shape_ori[1] - tmp_lab[::2]

        if random_landmark_mask is not None:
            if random() < random_landmark_mask:
                tmp_for_shape_choice = random()
                rand_color = [np.random.randint(10, 245) for _ in range(3)]
                rand_landmark = np.random.randint(0, int(len(tmp_lab) / 2))
                if tmp_for_shape_choice > 0.5:
                    rand_r = int(random() * min(shape_ori[:2]) / 12.0)
                    if rand_r == 0:
                        rand_r = 2
                    rand_x = int(tmp_lab[2*rand_landmark])
                    rand_y = int(tmp_lab[2*rand_landmark+1])
                    if not max([rand_x + rand_r, rand_y + rand_r]) > min(shape_ori):
                        cv2.circle(tmp_img, (rand_x, rand_y), rand_r, rand_color, -1)
                else:
                    rand_w = int(random() * (min(shape_ori[:2]) / 6.0))
                    if rand_w == 0:
                        rand_w = 4
                    rand_x = int(tmp_lab[2 * rand_landmark])
                    rand_y = int(tmp_lab[2 * rand_landmark + 1])
                    if not max([rand_x+rand_w, rand_y+rand_w]) > min(shape_ori):
                        cv2.rectangle(tmp_img, (rand_x, rand_y), (rand_x+rand_w, rand_y+rand_w), rand_color, -1)
        # if not test:
        #     tmp_img = (tmp_img - np.mean(tmp_img)) / np.std(tmp_img)
        return [np.array(tmp_img), np.array(tmp_lab)]
    return random_augmentation_func

random_aug_func = random_augmentation(scale=(0.4, 6),
                                      dirty_circle=0.3,
                                      random_flip_horizontal=None,
                                      random_filp_vertical=None,
                                      random_landmark_mask=0.3,
                                      random_squeeze=[0.4, 6],
                                      random_rotate=0.9)

def test_aug():

    func = random_augmentation(scale=(0.4, 6),
                              dirty_circle=0.3,
                              random_flip_horizontal=None,
                              random_filp_vertical=None,
                              random_landmark_mask=0.3,
                              random_squeeze=[0.4, 6],
                              random_rotate=0.7,
                               test=True)
    ds = dlib_file_list([r'C:\Users\Jackie\AppData\Roaming\feiq\Recv Files\25points_selected'])
    filenames = [ds[i][0] for i in range(len(ds))]
    labels = [ds[i][1] for i in range(len(ds))]
    for filename, label in zip(filenames, labels):
        tmp_p = os.path.join(r'E:\test\test_aug', filename.split('\\')[-1])
        tmp_p_ori = os.path.join(r'E:\test\test_aug', filename.split('\\')[-1])
        tmp_p_ori = tmp_p_ori[:-4] + '_ori.jpg'
        img = imread(filename)
        if img is None:
            continue
        l = len(label)
        img_ori = img.copy()
        for t in range(int(l / 2)):
            cv2.circle(img_ori, (int(round(label[2*t])), int(round(label[2*t + 1]))), 2, (0, 0, 255), -1)
        imwrite((tmp_p_ori), img_ori)
        img = img.astype(np.float32)
        img = img / 255.0
        label = np.array(label, dtype=np.float32)
        img, label = func([img, label])
        img = img * 255
        img = img.astype(np.uint8)
        for t in range(int(l / 2)):
            cv2.circle(img, (int(round(label[2*t])), int(round(label[2*t + 1]))), 2, (0, 0, 255), -1)
        imwrite((tmp_p), img)
        print("%s saved."%(tmp_p))


def distort_tf_function(image, label):
    """Perform random distortions on an image.
    Args:
    image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
    """
    with tf.name_scope("distort_color", values=[image]):
        image = tf.image.random_brightness(image, max_delta=0.1/255)
        image = tf.image.random_saturation(image, lower=0.015/255.0, upper=1.015/255)
        image = tf.image.random_hue(image, max_delta=0.001/255)
        image = tf.image.random_contrast(image, lower=0.015/255.0, upper=1.015/255.0)

    return image, label

def resize_img(img, label, shape):
    img.set_shape([None, None, None])
    img = tf.image.resize_images(img, shape)
    return img, label

def coord2imgs(label):
    label_zeros = np.zeros([40, 40, 21], dtype=np.float32)
    for idx in range(21):
        x = floor(40*label[2*idx])
        if x >= 40:
            x = 39
        y = floor(40*label[2*idx+1])
        if y >= 40:
            y = 39
        label_zeros[x, y, idx] = 1.0
    return label_zeros

def imgs2coord(label):
    l_list = []
    if label.ndim == 3:
        for idx in range(label.shape[-1]):
            max_idx = np.argmax(label[:, :, idx])
            l_list += [ max_idx % label.shape[1], floor(max_idx / label.shape[0]) ]
    elif label.ndim == 2:
        sort_idx = np.argsort(label, axis=None)[-21:]
        for idx in sort_idx[::-1]:
            l_list += [idx % label.shape[1], floor(idx / label.shape[0])]
    else:
        assert 0, "label.ndim must be equal to 3 or 2"
    l_list = np.array(l_list, dtype=np.float32)
    l_list[::2] = 1.0 * l_list[::2] / label.shape[1]
    l_list[1::2] = 1.0 * l_list[1::2] / label.shape[0]
    return l_list


def read_py_function(filename, label):
    img = imread(filename.decode())
    s = img.shape
    img = img.astype(np.float32)
    img = img / 255
    label = np.array(label, dtype=np.float32)
    img, label = random_aug_func([img, label])
    label[::2] = 1.0 * label[::2] / s[1]
    label[1::2] = 1.0 * label[1::2] / s[0]
    label = coord2imgs(label)
    return img.astype(np.float32), label.astype(np.float32)

def read_py_function_no_aug(filename, label):
    img = imread(filename.decode())
    s = img.shape
    img = img.astype(np.float32)
    img = img / 255
    label = np.array(label, dtype=np.float32)
    label[::2] = 1.0 * label[::2] / s[1]
    label[1::2] = 1.0 * label[1::2] / s[0]
    label = coord2imgs(label)
    return img.astype(np.float32), label.astype(np.float32)


def read_fn_0(filename, labels, config=None):
    f = tf.constant(filename)
    l = tf.constant(labels)

    dataset = tf.contrib.data.Dataset.from_tensor_slices((f, l))

    # dataset = dataset.map(
    #     lambda f,l: tf.py_func(read_py_func, [f, l], [tf.float32, tf.float32]))
    # dataset = dataset.map(lambda f, l: _preprocess_tf_func(f, l, config), num_thread=32)
    return dataset

def get_dataset_from_file(config=None):
    total_ds = dlib_file_list(path)
    np.random.shuffle(total_ds)
    split_n = ceil(len(total_ds) * config.train_ratio)
    train_ds = total_ds[:split_n]
    test_ds = total_ds[split_n:]

    info = 'Numbers of train set: %d\n Numbers of test set: %d\n' % (len(train_ds), len(test_ds))

    def _f(ds, is_train=True):
        filenames = [ds[i][0] for i in range(len(ds))]
        labels = [ds[i][1] for i in range(len(ds))]
        dataset = read_fn_0(filenames, labels)

        if is_train:
            dataset = dataset.map(
                lambda filename, label: tuple(tf.py_func(
                    read_py_function, [filename, label], [tf.float32, tf.float32])), num_threads=256)
        else:
            dataset = dataset.map(
                lambda filename, label: tuple(tf.py_func(
                    read_py_function_no_aug, [filename, label], [tf.float32, tf.float32])), num_threads=128)
        # dataset = dataset.map(distort_tf_function, num_threads=64)
        dataset = dataset.map(lambda i, l: resize_img(i, l, [config.width, config.height]), num_threads=64)
        dataset = dataset.shuffle(buffer_size=config.shuffle_buffer_size)
        dataset = dataset.batch(config.batch_size)
        return dataset
    return _f(train_ds), _f(test_ds, False), info

def get_dataset_from_h5(config):
    """Use tensorpack.dataflow.HDF5Data to get dataset through hdf5 file.

    Warning:
        This will load all data into memory.
    """
    train = MyHDF5Dataset(config.train_path, config.data_paths)
    test = MyHDF5Dataset(config.test_path, config.data_paths)

    info = 'Numbers of train set: %d\n Numbers of test set: %d\n' % (len(train), len(test))

    train_ds = train.data
    test_ds = test.data

    train_ds = read_fn(train_ds[0], train_ds[1], config)
    test_ds = read_fn(test_ds[0], test_ds[1], config)

    return train_ds, test_ds, info

def test_dataset_save_img():
    # train_ds = dlib_file_list()
    # np.random.shuffle(train_ds)
    # filenames = [train_ds[i][0] for i in range(len(train_ds))]
    # labels = [train_ds[i][1] for i in range(len(train_ds))]
    # dataset = read_fn_0(filenames, labels)
    # dataset = dataset.map(
    #     lambda filename, label: tuple(tf.py_func(
    #         read_py_function, [filename, label], [tf.float32, tf.float32])), num_threads=256)
    # # dataset = dataset.map(distort_tf_function)
    # dataset = dataset.map(lambda i, l: resize_img(i, l, [40, 40]), num_threads=256)
    # dataset = dataset.shuffle(buffer_size=1000)
    # dataset = dataset.batch(32)
    model_config = ModelConfig()
    model_config.width = 40
    model_config.height = 40
    model_config.channels = 3
    model_config.train_ratio = 0.8
    model_config.shuffle_buffer_size = 512
    model_config.batch_size = 32

    train_ds, test_ds, info = get_dataset_from_file(model_config)
    dataset = train_ds
    test_result = test_dataset(dataset)
    print(len(test_result))
    print(len(test_result[0]))
    img = test_result[0][0]
    s = img.shape
    img = img * 255
    img = img.astype(np.uint8)
    print(s)
    label = test_result[1][0]
    print(label.shape)
    l = len(label)
    for t in range(int(l/2)):
        cv2.circle(img, (int(s[0]*label[2 * t]), int(s[1]*label[2*t+1])), 2, (0, 0, 255), -1)
    imwrite(('test.jpg'), img)


